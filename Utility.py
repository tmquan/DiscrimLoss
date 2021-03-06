#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob, time

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
import malis

# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger

# Tensorflow 
import tensorflow as tf

# Tensorlayer
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe

# Augmentation
import Augmentor

###############################################################################
NB_FILTERS = 32
###############################################################################
def time_seed ():
	seed = None
	while seed == None:
		cur_time = time.time ()
		seed = int ((cur_time - int (cur_time)) * 1000000)
		return seed


###############################################################################
def INReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.leaky_relu(x, name=name)
	
def BNLReLU(x, name=None):
	x = BatchNorm('bn', x)
	return tf.nn.leaky_relu(x, name=name)

###############################################################################
def seg_to_aff_op(seg, nhood=tf.constant(malis.mknhood3d(1)), name='SegToAff'):
	# Squeeze the segmentation to 3D
	seg = tf.squeeze(seg, axis=-1)
	# Define the numpy function to transform segmentation to affinity graph
	np_func = lambda seg, nhood: malis.seg_to_affgraph (seg.astype(np.int32), nhood).astype(np.float32)
	# Convert the numpy function to tensorflow function
	tf_func = tf.py_func(np_func, [tf.cast(seg, tf.int32), nhood], [tf.float32], name=name)
	# Reshape the result, notice that layout format from malis is 3, dimx, dimy, dimx
	ret = tf.reshape(tf_func[0], [3, seg.shape[0], seg.shape[1], seg.shape[2]])
	# Transpose the result so that the dimension 3 go to the last channel
	ret = tf.transpose(ret, [1, 2, 3, 0])
	# print ret.get_shape().as_list()
	return ret
###############################################################################
def aff_to_seg_op(aff, nhood=tf.constant(malis.mknhood3d(1)), threshold=tf.constant(np.array([0.5])), name='AffToSeg'):
	# Define the numpy function to transform affinity to segmentation
	def np_func (aff, nhood, threshold):
		aff = np.transpose(aff, [3, 0, 1, 2]) # zyx3 to 3zyx
		ret = malis.connected_components_affgraph((aff > threshold[0]).astype(np.int32), nhood)[0].astype(np.int32) 
		ret = skimage.measure.label(ret).astype(np.int32)
		return ret
	# print aff.get_shape().as_list()
	# Convert numpy function to tensorflow function
	tf_func = tf.py_func(np_func, [aff, nhood, threshold], [tf.int32], name=name)
	ret = tf.reshape(tf_func[0], [aff.shape[0], aff.shape[1], aff.shape[2]])
	ret = tf.expand_dims(ret, axis=-1)
	# print ret.get_shape().as_list()
	return ret
###############################################################################
def toMaxLabels(label, factor=320):
	result = tf.cast(label, tf.float32)
	status = tf.equal(result, -1.0*tf.ones_like(result))
	result = tf.where(status, tf.zeros_like(result), result, name='removedBackground') # From -1 to 0
	result = result * factor # From 0~1 to 0~MAXLABEL
	result = tf.round(result)
	return tf.cast(result, tf.int32)
###############################################################################
# Utility function for scaling 
def tf_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / maxVal - 0.5) * 2.0
###############################################################################
def tf_2imag(x, maxVal = 255.0, name='ToRangeImag'):
	with tf.variable_scope(name):
		return (x / 2.0 + 0.5) * maxVal

# Utility function for scaling 
def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	return (x / maxVal - 0.5) * 2.0
###############################################################################
def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
	return (x / 2.0 + 0.5) * maxVal


###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False):
	with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=3):
		input = x
		return (LinearWrap(x)
				.Conv2D('conv1', chan, padding='SAME', dilation_rate=1)
				.Conv2D('conv2', chan, padding='SAME', dilation_rate=2)
				.Conv2D('conv4', chan, padding='SAME', dilation_rate=4)             
				.Conv2D('conv5', chan, padding='SAME', dilation_rate=8)
				.Conv2D('conv0', chan, padding='SAME', nl=tf.identity)
				.InstanceNorm('inorm')()) + input

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=2, stride=1):
	with argscope([Conv2D], nl=INLReLU, stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		old_shape = inputs.get_shape().as_list()
		# results = tf.reshape(results, [-1, chan, old_shape[2]*scale, old_shape[3]*scale])
		# results = tf.reshape(results, [-1, old_shape[1]*scale, old_shape[2]*scale, chan])
		if scale>1:
			results = tf.depth_to_space(results, scale, name='depth2space', data_format='NHWC')
		return results

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
		x = (LinearWrap(x)
			# .Dropout('drop', 0.75)
			.Conv2D('conv_i', chan, stride=2) 
			.residual('res_', chan, first=True)
			.Conv2D('conv_o', chan, stride=1) 
			())
		return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
				
		x = (LinearWrap(x)
			.Subpix2D('deconv_i', chan, scale=1) 
			.residual('res2_', chan, first=True)
			.Subpix2D('deconv_o', chan, scale=2) 
			# .Dropout('drop', 0.75)
			())
		return x

###############################################################################
@auto_reuse_variable_scope
def arch_generator(img, last_dim=1):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)
		# e3 = Dropout('dr', e3, 0.5)

		d3 = residual_dec('d3',    e3, NB_FILTERS*4)
		d2 = residual_dec('d2', d3+e2, NB_FILTERS*2)
		d1 = residual_dec('d1', d2+e1, NB_FILTERS*1)
		d0 = residual_dec('d0', d1+e0, NB_FILTERS*1) 
		dd =  (LinearWrap(d0)
				.Conv2D('convlast', last_dim, kernel_shape=3, stride=1, padding='SAME', nl=tf.tanh, use_bias=True) ())
		return dd, d0

@auto_reuse_variable_scope
def arch_discriminator(img):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		img = Conv2D('conv0', img, NB_FILTERS, nl=tf.nn.leaky_relu)
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		# e0 = Dropout('dr', e0, 0.5)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)

		ret = Conv2D('convlast', e3, 1, stride=1, padding='SAME', nl=tf.identity, use_bias=True)
		return ret

# Tiramis
def batch_norm(x, training, name):
	with tf.variable_scope(name):
		x = tf.cond(training, lambda: tf.contrib.layers.batch_norm(x, is_training=True, scope=name+'_batch_norm'),
					lambda: tf.contrib.layers.batch_norm(x, is_training=False, scope=name+'_batch_norm', reuse=True))
	return x

def conv_layer(x, training, filters, name):
	with tf.name_scope(name):
		x = self.batch_norm(x, training, name=name+'_bn')
		x = tf.nn.relu(x, name=name+'_relu')
		x = tf.layers.conv2d(x,
							 filters=filters,
							 kernel_size=[3, 3],
							 strides=[1, 1],
							 padding='SAME',
							 dilation_rate=[1, 1],
							 activation=None,
							 kernel_initializer=tf.contrib.layers.xavier_initializer(),
							 name=name+'_conv3x3')
		x = tf.layers.dropout(x, rate=0.2, training=training, name=name+'_dropout')

	return x

def dense_block(x, training, block_nb, name):
	dense_out = []
	with tf.name_scope(name):
		for i in range(self.layers_per_block[block_nb]):
			conv = self.conv_layer(x, training, self.growth_k, name=name+'_layer_'+str(i))
			x = tf.concat([conv, x], axis=3)
			dense_out.append(conv)

		x = tf.concat(dense_out, axis=3)

	return x

def transition_down(x, training, filters, name):
	with tf.name_scope(name):
		x = self.batch_norm(x, training, name=name+'_bn')
		x = tf.nn.relu(x, name=name+'relu')
		x = tf.layers.conv2d(x,
							 filters=filters,
							 kernel_size=[1, 1],
							 strides=[1, 1],
							 padding='SAME',
							 dilation_rate=[1, 1],
							 activation=None,
							 kernel_initializer=tf.contrib.layers.xavier_initializer(),
							 name=name+'_conv1x1')
		x = tf.layers.dropout(x, rate=0.2, training=training, name=name+'_dropout')
		x = tf.nn.max_pool(x, [1, 4, 4, 1], [1, 2, 2, 1], padding='SAME', name=name+'_maxpool2x2')

	return x

def transition_up(x, filters, name):
	with tf.name_scope(name):
		x = tf.layers.conv2d_transpose(x,
									   filters=filters,
									   kernel_size=[3, 3],
									   strides=[2, 2],
									   padding='SAME',
									   activation=None,
									   kernel_initializer=tf.contrib.layers.xavier_initializer(),
									   name=name+'_trans_conv3x3')

	return x

@auto_reuse_variable_scope
def arch_tiramisu(img, growth_k=16, layers_per_block=(2,2), last_dim=1):
	concats = []
	nb_blocks = len(layers_per_block)
	x = tf.layers.conv2d(x,
						 filters=32,
						 kernel_size=[3, 3],
						 strides=[1, 1],
						 padding='SAME',
						 dilation_rate=[1, 1],
						 activation=None,
						 kernel_initializer=tf.contrib.layers.xavier_initializer(),
						 name='first_conv3x3')
	print(x.get_shape())
	print("Building downsample path...")
	for block_nb in range(0, nb_blocks):
		dense = dense_block(x, training, block_nb, 'down_dense_block_' + str(block_nb))

		if block_nb != nb_blocks - 1:
			x = tf.concat([x, dense], axis=3, name='down_concat_' + str(block_nb))
			print(x.get_shape())
			concats.append(x)
			x = transition_down(x, training, x.get_shape()[-1], 'trans_down_' + str(block_nb))

	print(dense.get_shape())
	print("Building upsample path...")
	for i, block_nb in enumerate(range(nb_blocks - 1, 0, -1)):
		x = transition_up(x, x.get_shape()[-1], 'trans_up_' + str(block_nb))
		x = tf.concat([x, concats[len(concats) - i - 1]], axis=3, name='up_concat_' + str(block_nb))
		print(x.get_shape())
		x = dense_block(x, training, block_nb, 'up_dense_block_' + str(block_nb))

	x = tf.layers.conv2d(x,
						 filters=num_classes,
						 kernel_size=[1, 1],
						 strides=[1, 1],
						 padding='SAME',
						 dilation_rate=[1, 1],
						 activation=None,
						 kernel_initializer=tf.contrib.layers.xavier_initializer(),
						 name='last_conv1x1')
	print(x.get_shape())

	return x

###########




def supervised_clustering_loss(prediction, correct_label, feature_dim, label_shape):
	Y = tf.reshape(correct_label, [label_shape[1]*label_shape[0], 1])
	F = tf.reshape(prediction, [label_shape[1]*label_shape[0], feature_dim])
	diagy = tf.reduce_sum(Y,0)
	onesy = tf.ones(diagy.get_shape())
	J = tf.matmul(Y,tf.diag(tf.rsqrt(tf.where(tf.greater_equal(diagy,onesy),diagy,onesy))))
	[S,U,V] = tf.svd(F)
	Slength = tf.cast(tf.reduce_max(S.get_shape()), tf.float32)
	maxS = tf.fill(tf.shape(S),tf.scalar_mul(tf.scalar_mul(1e-15,tf.reduce_max(S)),Slength))
	ST = tf.where(tf.greater_equal(S,maxS),tf.div(tf.ones(S.get_shape()),S),tf.zeros(S.get_shape()))
	pinvF = tf.transpose(tf.matmul(U,tf.matmul(tf.diag(ST),V,False,True)))
	FJ = tf.matmul(pinvF,J)
	G = tf.matmul(tf.subtract(tf.matmul(F,FJ),J),FJ,False,True)
	loss = tf.reduce_sum(tf.multiply(tf.stop_gradient(G),F))
	return loss



def tf_norm(inputs, axis=1, epsilon=1e-7,  name='safe_norm'):
	squared_norm    = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
	safe_norm       = tf.sqrt(squared_norm+epsilon)
	return tf.identity(safe_norm, name=name)

def discriminative_loss_single(prediction, correct_label, feature_dim, label_shape, 
							delta_v, delta_d, param_var, param_dist, param_reg):
	
	''' Discriminative loss for a single prediction/label pair.
	:param prediction: inference of network
	:param correct_label: instance label
	:feature_dim: feature dimension of prediction
	:param label_shape: shape of label
	:param delta_v: cutoff variance distance
	:param delta_d: curoff cluster distance
	:param param_var: weight for intra cluster variance
	:param param_dist: weight for inter cluster distances
	:param param_reg: weight regularization
	'''

	### Reshape so pixels are aligned along a vector
	correct_label = tf.reshape(correct_label, [label_shape[1]*label_shape[0]])
	reshaped_pred = tf.reshape(prediction, [label_shape[1]*label_shape[0], feature_dim])

	### Count instances
	unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
	counts = tf.cast(counts, tf.float32)
	num_instances = tf.size(unique_labels)

	segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

	mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
	mu_expand = tf.gather(mu, unique_id)

	### Calculate l_var
	distance = tf_norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
	distance = tf.subtract(distance, delta_v)
	distance = tf.clip_by_value(distance, 0., distance)
	distance = tf.square(distance)

	l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
	l_var = tf.div(l_var, counts)
	l_var = tf.reduce_sum(l_var)
	l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))
	
	### Calculate l_dist
	
	# Get distance for each pair of clusters like this:
	#   mu_1 - mu_1
	#   mu_2 - mu_1
	#   mu_3 - mu_1
	#   mu_1 - mu_2
	#   mu_2 - mu_2
	#   mu_3 - mu_2
	#   mu_1 - mu_3
	#   mu_2 - mu_3
	#   mu_3 - mu_3

	mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
	mu_band_rep = tf.tile(mu, [1, num_instances])
	mu_band_rep = tf.reshape(mu_band_rep, (num_instances*num_instances, feature_dim))

	mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)
	
	# Filter out zeros from same cluster subtraction
	intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
	zero_vector = tf.zeros(1, dtype=tf.float32)
	bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
	mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

	mu_norm = tf_norm(mu_diff_bool, axis=1)
	mu_norm = tf.subtract(2.*delta_d, mu_norm)
	mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
	mu_norm = tf.square(mu_norm)

	l_dist = tf.reduce_mean(mu_norm)

	### Calculate l_reg
	l_reg = tf.reduce_mean(tf_norm(mu, axis=1))

	param_scale = 1.
	l_var = param_var * l_var
	l_dist = param_dist * l_dist
	l_reg = param_reg * l_reg

	loss = param_scale*(l_var + l_dist + l_reg)
	
	return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, image_shape, 
				delta_v, delta_d, param_var, param_dist, param_reg):
	''' Iterate over a batch of prediction/label and cumulate loss
	:return: discriminative loss and its three components
	'''
	def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
		return tf.less(i, tf.shape(batch)[0])

	def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
		disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim, image_shape, 
						delta_v, delta_d, param_var, param_dist, param_reg)

		out_loss = out_loss.write(i, disc_loss)
		out_var = out_var.write(i, l_var)
		out_dist = out_dist.write(i, l_dist)
		out_reg = out_reg.write(i, l_reg)

		return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

	# TensorArray is a data structure that support dynamic writing
	output_ta_loss = tf.TensorArray(dtype=tf.float32,
				   size=0,
				   dynamic_size=True)
	output_ta_var = tf.TensorArray(dtype=tf.float32,
				   size=0,
				   dynamic_size=True)
	output_ta_dist = tf.TensorArray(dtype=tf.float32,
				   size=0,
				   dynamic_size=True)
	output_ta_reg = tf.TensorArray(dtype=tf.float32,
				   size=0,
				   dynamic_size=True)

	_, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _  = tf.while_loop(cond, body, [correct_label, 
														prediction, 
														output_ta_loss, 
														output_ta_var, 
														output_ta_dist, 
														output_ta_reg, 
														0])
	out_loss_op = out_loss_op.stack()
	out_var_op = out_var_op.stack()
	out_dist_op = out_dist_op.stack()
	out_reg_op = out_reg_op.stack()
	
	disc_loss = tf.reduce_mean(out_loss_op)
	l_var = tf.reduce_mean(out_var_op)
	l_dist = tf.reduce_mean(out_dist_op)
	l_reg = tf.reduce_mean(out_reg_op)

	return disc_loss, l_var, l_dist, l_reg


## Implement clustering algorithm here
from sklearn import cluster



def tf_cluster_dbscan(X, feature_dim, label_shape=None, eps=0.3, name='DBSCAN'):
	# Define the numpy function to perform such a clustering the high dimensional feature
	def np_func(X, feature_dim, label_shape=None, eps=0.3):
		# Perform clustering on high dimensional channel image
		feats_shape = X.shape
		# if label_shape==None:
		# 	label_shape = feats_shape[:-1]
		# 	label_shape = np.expand_dims(label_shape, -1)
		# Flatten the 
		X_flatten = np.reshape(X, [-1, feature_dim])
		algorithm = cluster.DBSCAN(eps=eps)
		algorithm.fit(X_flatten)
		# Get the result in float32
		y_pred_flatten = algorithm.labels_.astype(np.float32)
		y_pred = np.reshape(y_pred_flatten, label_shape)
		return y_pred

	# print aff.get_shape().as_list()
	# Convert numpy function to tensorflow function
	tf_func = tf.py_func(np_func, [X, feature_dim, label_shape, eps], [tf.float32], name=name)
	ret = tf.reshape(tf_func[0], label_shape)
	print(X)
	print(ret)
	return ret