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


class CVPPPImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32', isTrain=False, isValid=False, isTest=False):
		self.dtype      = dtype
		self.imageDir   = imageDir
		self.labelDir   = labelDir
		self._size      = size
		self.isTrain    = isTrain
		self.isValid    = isValid

		images = natsorted (glob.glob(self.imageDir + '/*rgb.png'))
		labels = natsorted (glob.glob(self.labelDir + '/*label.png'))
		self.images = []
		self.labels = []
		self.data_seed = time_seed ()
		self.data_rand = np.random.RandomState(self.data_seed)
		self.rng = np.random.RandomState(999)
		for i in range (len (images)):
			image = images[i]
			label = labels[i]
		self.images.append (skimage.io.imread (image))
		self.labels.append (skimage.io.imread (label))
		