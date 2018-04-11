from Utility import * 

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

# Sklearn
from sklearn.metrics.cluster import adjusted_rand_score
###############################################################################
MAX_LABEL=320
DIMZ = 1
DIMY = 512
DIMX = 512

###############################################################################
class CVPPPImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32', isTrain=False, isValid=False, isTest=False):
		self.dtype      = dtype
		self.imageDir   = imageDir
		self.labelDir   = labelDir
		self._size      = size
		self.isTrain    = isTrain
		self.isValid    = isValid

		imageFiles = natsorted (glob.glob(self.imageDir + '/*rgb.png'))
		labelFiles = natsorted (glob.glob(self.labelDir + '/*label.png'))
		self.images = []
		self.labels = []
		self.data_seed = time_seed ()
		self.data_rand = np.random.RandomState(self.data_seed)
		self.rng = np.random.RandomState(999)
		for i in range (len (imageFiles)):
			imageFile = imageFiles[i]
			labelFile = labelFiles[i]
			image = skimage.io.imread(imageFile)
			label = skimage.io.imread(labelFile)
			image = skimage.transform.resize(image, output_shape=(DIMY, DIMX, 3), order=0, preserve_range=True)
			label = skimage.transform.resize(label, output_shape=(DIMY, DIMX, 3), order=0, preserve_range=True)
			
			image = skimage.color.rgb2gray(image)
			label = skimage.color.rgb2gray(label)

			# image = np.expand_dims(image, axis=-1)
			# label = np.expand_dims(label, axis=-1)
			
			# image = np.expand_dims(image, axis=0)
			# label = np.expand_dims(label, axis=0)

			self.images.append (image)
			self.labels.append (label)
		

	def size(self):
		return self._size

	def get_data(self):
		for k in range(self._size):
			#
			# Pick randomly a tuple of training instance
			#
			rand_index = self.data_rand.randint(0, len(self.images))
			image_p = self.images[rand_index].copy ()
			label_p = self.labels[rand_index].copy ()
			
			# Declare augmentation here
			p = Augmentor.Pipeline()
			# p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
			# p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5)
			# p.zoom_random(probability=0.5, percentage_area=0.8)
			p.rotate_random_90(probability=0.75)
			p.flip_left_right(probability=0.5)
			p.flip_top_bottom(probability=0.5)

			seed = time_seed ()

			if self.isTrain:
				# Augment the pair image for same seed
				# p.set_seed(seed)
				# image = p._execute_with_array(image_p.copy())
				p.set_seed(seed)
				label = p._execute_with_array(label_p.copy())
				# label = label_p.copy()
			else:
				label = label_p.copy()

			label = skimage.measure.label(label)*10
			# label = skimage.measure.label(label)
			image = 255.0*(1-skimage.segmentation.find_boundaries(label, mode='thick'))
			

			image = np.expand_dims(image, axis=-1)
			label = np.expand_dims(label, axis=-1)
			
			image = np.expand_dims(image, axis=0)
			label = np.expand_dims(label, axis=0)

			yield [image.astype(np.float32), 
				   label.astype(np.float32), ]


class Model(ModelDesc):
	#FusionNet
	@auto_reuse_variable_scope
	def generator(self, img, last_dim=1):
		assert img is not None
		return arch_generator(img, last_dim=last_dim)
		# return arch_fusionnet(img)

	@auto_reuse_variable_scope
	def discriminator(self, img):
		assert img is not None
		return arch_discriminator(img)


	def inputs(self):
		return [
			tf.placeholder(tf.float32, (1, DIMY, DIMX, 1), 'image'),
			tf.placeholder(tf.float32, (1, DIMY, DIMX, 1), 'label'),
			]

	def build_graph(self, image, label):
		pi, pl = image, label

		pi = tf_2tanh(pi)
		pl = tf_2tanh(pl)


		with tf.variable_scope('gen'):
			# with tf.device('/device:GPU:0'):
				with tf.variable_scope('feats'):
					pid, _  = self.generator(pi, last_dim=16)
			# with tf.device('/device:GPU:1'):
				with tf.variable_scope('label'):
					pil, _  = self.generator(pid, last_dim=1)


		losses = []

		

		pa   = seg_to_aff_op(tf_2imag(pl)+1.0,  name='pa')		# 0 1
		pila = seg_to_aff_op(tf_2imag(pil)+1.0, name='pila')		# 0 1

		with tf.name_scope('loss_spectral'):
			spectral_loss  = supervised_clustering_loss(tf.concat([tf_2imag(pid)/255.0, pil/255.0, pila], axis=-1), 
																	 tf_2imag(pl), 
																	 20,
																	 (DIMY, DIMX), 
																	)

			losses.append(1e1*spectral_loss)
			add_moving_summary(spectral_loss)

		with tf.name_scope('loss_discrim'):
			param_var 	= 1.0 #args.var
			param_dist 	= 1.0 #args.dist
			param_reg 	= 0.001 #args.reg
			delta_v 	= 0.5 #args.dvar
			delta_d 	= 1.5 #args.ddist

			#discrim_loss  =  ### Optimization operations
			discrim_loss, l_var, l_dist, l_reg = discriminative_loss(tf.concat([tf_2imag(pid)/255.0, pil/255.0, pila], axis=-1), 
																	 tf_2imag(pl), 
																	 20, 
																	 (DIMY, DIMX), 
																     delta_v, delta_d, param_var, param_dist, param_reg)

			losses.append(1e1*discrim_loss)
			add_moving_summary(discrim_loss)

		with tf.name_scope('loss_aff'):		
			aff_ila = tf.identity(tf.subtract(binary_cross_entropy(pa, pila), 
					    		 			  dice_coe(pa, pila, axis=[0,1,2,3], loss_type='jaccard')),
								 name='aff_ila')			
			#losses.append(3e-3*aff_ila)
			add_moving_summary(aff_ila)

		with tf.name_scope('loss_smooth'):		
			smooth_ila = tf.reduce_mean((tf.ones_like(pila) - pila), name='smooth_ila')			
			losses.append(1e1*smooth_ila)
			add_moving_summary(smooth_ila)

		with tf.name_scope('loss_mae'):
			mae_il  = tf.reduce_mean(tf.abs(pl - pil), name='mae_il')
			losses.append(1e0*mae_il)
			add_moving_summary(mae_il)
			
			mae_ila = tf.reduce_mean(tf.abs(pa - pila), name='mae_ila')
			losses.append(1e0*mae_ila)
			add_moving_summary(mae_ila)
			

		self.cost = tf.reduce_sum(losses, name='self.cost')
		add_moving_summary(self.cost)
		# Visualization

		# Segmentation
		pz = tf.zeros_like(pi)
		# viz = tf.concat([image, label, pic], axis=2)
		viz = tf.concat([tf.concat([pi, pl, pil], axis=2),
						 tf.concat([pa[...,0:1], pa[...,1:2], pa[...,2:3]], axis=2),
						 tf.concat([pila[...,0:1], pila[...,1:2], pila[...,2:3]], axis=2),
						 ], axis=1)
		viz = tf_2imag(viz)

		viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		tf.summary.image('colorized', viz, max_outputs=50)


	def optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

###############################################################################
class VisualizeRunner(Callback):
	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			['image', 'label'], ['viz'])

	def _before_train(self):
		global args
		self.test_ds = get_data(args.data, isTrain=False, isValid=False, isTest=True)

	def _trigger(self):
		for lst in self.test_ds.get_data():
			viz_test = self.pred(lst)
			viz_test = np.squeeze(np.array(viz_test))

			#print viz_test.shape

			self.trainer.monitors.put_image('viz_test', viz_test)
###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False):
	# Process the directories 
	if isTrain:
		num=100
		names = ['trainA', 'trainB']
	if isValid:
		num=1
		names = ['trainA', 'trainB']
	if isTest:
		num=10
		names = ['validA', 'validB']

	
	dset  = CVPPPImageDataFlow(os.path.join(dataDir, names[0]),
					 		   os.path.join(dataDir, names[1]),
							   num, 
							   isTrain=isTrain, 
							   isValid=isValid, 
							   isTest =isTest)
	dset.reset_state()
	return dset
###############################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu',        default='0', help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--data',  default='data/Kasthuri15/3D/', required=True, 
									help='Data directory, contain trainA/trainB/validA/validB')
	parser.add_argument('--load',   help='Load the model path')
	parser.add_argument('--sample', help='Run the deployment on an instance',
									action='store_true')

	args = parser.parse_args()
	# python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

	
	train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False)
	valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False)
	# test_ds  = get_data(args.data, isTrain=False, isValid=False, isTest=True)


	train_ds  = PrefetchDataZMQ(train_ds, 4)
	train_ds  = PrintData(train_ds)
	# train_ds  = QueueInput(train_ds)
	model     = Model()

	os.environ['PYTHONWARNINGS'] = 'ignore'

	# Set the GPU
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	# Running train or deploy
	if args.sample:
		# TODO
		# sample
		pass
	else:
		# Set up configuration
		# Set the logger directory
		logger.auto_set_dir()

		# session_init = SaverRestore(args.load) if args.load else None, 

		# Set up configuration
		config = TrainConfig(
			model           =   model, 
			dataflow        =   train_ds,
			callbacks       =   [
				PeriodicTrigger(ModelSaver(), every_k_epochs=50),
				PeriodicTrigger(VisualizeRunner(), every_k_epochs=5),
				PeriodicTrigger(InferenceRunner(valid_ds, [ScalarStats('loss_mae/mae_il')]), every_k_epochs=1),
				# ScheduledHyperParamSetter('learning_rate', [(0, 1e-6), (300, 1e-6)], interp='linear')
				ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
				# ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
				# HumanHyperParamSetter('learning_rate'),
				],
			max_epoch       =   500, 
			session_init    =    SaverRestore(args.load) if args.load else None,
			#nr_tower        =   max(get_nr_gpu(), 1)
			)
	
		# Train the model
		# SyncMultiGPUTrainer(config).train()
		# trainer = SyncMultiGPUTrainerReplicated(max(get_nr_gpu(), 1))
		launch_train_with_config(config, QueueInputTrainer())