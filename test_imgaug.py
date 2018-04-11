import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
images = np.array(
	[ia.quokka(size=(64, 64)) for _ in range(32)],
	dtype=np.uint8
)

images = np.squeeze(images[...,0])
images = np.transpose(images, [1,2,0])
labels = 255-images

    
seq = iaa.Sequential([
	#iaa.Fliplr(0.5), # horizontal flips
	iaa.Crop(percent=(0, 0.2)), # random crops
	# Small gaussian blur with random sigma between 0 and 0.5.
	# But we only blur about 50% of all images.
	iaa.Sometimes(0.5,
		iaa.GaussianBlur(sigma=(0, 0.5))
	),
	# Strengthen or weaken the contrast in each image.
	iaa.ContrastNormalization((0.75, 1.5)),
	# Add gaussian noise.
	# For 50% of all images, we sample the noise once per pixel.
	# For the other 50% of all images, we sample the noise per pixel AND
	# channel. This can change the color (not only brightness) of the
	# pixels.
	iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
	# Make some images brighter and some darker.
	# In 20% of all cases, we sample the multiplier once per channel,
	# which can end up changing the color of the images.
	iaa.Multiply((0.8, 1.2), per_channel=0.2),
	# Apply affine transformations to each image.
	# Scale/zoom them, translate/move them, rotate them and shear them.
	iaa.Affine(
		scale={"x": (1.0, 1.5), "y": (1.0, 1.5)},
#		translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#		rotate=(-25, 25),
#		shear=(-8, 8)
	)
    iaa.ElasticTransformation(alpha=(0.5, 8.0), sigma=1.0, cval=0, order=0)),
], random_order=True) # apply augmenters in random order

#images_aug = seq.augment_images(images)
_aug = seq._to_deterministic()
new_images = _aug.augment_image(images)
new_labels = _aug.augment_image(labels)
for i in range(4):
    #plt.imshow(images[i])
    plt.imshow(new_images[...,i])
    plt.show()

for i in range(4):
    plt.imshow(new_labels[...,i])
    plt.show()