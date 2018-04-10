import numpy as np 
import Augmentor
import glob
import os
from natsort import natsorted # natural sort file name
import skimage.io 
import skimage.transform 

import cv2 
import matplotlib.pyplot as plt 

DIMY = 512
DIMX = 512

imageDir = 'data/CVPPP2017/trainA/*.png'
labelDir = 'data/CVPPP2017/trainB/*.png'
imageFiles = natsorted(glob.glob(imageDir))
labelFiles = natsorted(glob.glob(labelDir))

print(imageFiles[:10])
print(labelFiles[:10])

images = []
labels = []

for imageFile, labelFile in zip(imageFiles, labelFiles):
	# image = cv2.imread(imageFile, cv2.IMREAD_COLOR)
	# label = cv2.imread(labelFile, cv2.IMREAD_COLOR)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
	# image = cv2.resize(image, (DIMY, DIMX), interpolation=cv2.INTER_NEAREST) 
	# label = cv2.resize(label, (DIMY, DIMX), interpolation=cv2.INTER_NEAREST) 

	image = skimage.io.imread(imageFile)
	label = skimage.io.imread(labelFile)
	image = skimage.transform.resize(image, output_shape=(DIMY, DIMX, 3), order=0, preserve_range=True)
	label = skimage.transform.resize(label, output_shape=(DIMY, DIMX, 3), order=0, preserve_range=True)
	# print image.shape
	# print label.shape
	images.append(image)
	labels.append(label)

images = np.array(images).astype(np.uint8)
labels = np.array(labels).astype(np.uint8)

print(images.shape)
print(labels.shape)




###############################################################

p = Augmentor.Pipeline()
p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
p.random_distortion(probability=1, grid_width=8, grid_height=8, magnitude=10)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
idx = 1
for idx in range(5):
	p.set_seed(idx)
	img = p._execute_with_array(images[idx])
	p.set_seed(idx)
	lbl = p._execute_with_array(labels[idx])
	plt.imshow(np.concatenate([images[idx], img, labels[idx], lbl], axis=1))
	plt.show()