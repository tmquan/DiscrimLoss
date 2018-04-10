import Augmentor 
import matplotlib.pyplot as plt
import numpy as np 
import os

# Create your new operation by inheriting from the Operation superclass:
class PipelineGroundtruth(Augmentor.Pipeline):
	# Some class variables we use often
	_probability_error_text = "The probability argument must be between 0 and 1."
	_threshold_error_text = "The value of threshold must be between 0 and 255."
	_valid_formats = ["PNG", "BMP", "TIF", "JPEG"]
	_legal_filters = ["NEAREST", "BICUBIC", "ANTIALIAS", "BILINEAR"]

	def __init__(self, source_directory=None, output_directory="output", save_format=None):
		"""
		Create a new Pipeline object pointing to a directory containing your
		original image dataset.

		Create a new Pipeline object, using the :attr:`source_directory`
		parameter as a source directory where your original images are
		stored. This folder will be scanned, and any valid file files
		will be collected and used as the original dataset that should
		be augmented. The scan will find any image files with the extensions
		JPEG/JPG, PNG, and GIF (case insensitive).

		:param source_directory: A directory on your filesystem where your
		 original images are stored.
		:param output_directory: Specifies where augmented images should be
		 saved to the disk. Default is the directory **output** relative to
		 the path where the original image set was specified. If it does not
		 exist it will be created.
		:param save_format: The file format to use when saving newly created,
		 augmented images. Default is JPEG. Legal options are BMP, PNG, and
		 GIF.
		:return: A :class:`Pipeline` object.
		"""
		import random
		random.seed()

		# TODO: Allow a single image to be added when initialising.
		# Initialise some variables for the Pipeline object.
		self.image_counter = 0
		self.augmentor_images = []
		self.distinct_dimensions = set()
		self.distinct_formats = set()
		self.save_format = save_format
		self.operations = []
		self.class_labels = []
		self.process_ground_truth_images = False

		# Now we populate some fields, which we may need to do again later if another
		# directory is added, so we place it all in a function of its own.
		if source_directory is not None:
			self._populate(source_directory=source_directory,
						   output_directory=output_directory,
						   ground_truth_directory=None,
						   ground_truth_output_directory=output_directory)
	def ground_truth(self, ground_truth_directory, del_str=None, add_str=None):
		"""
		Specifies a directory containing corresponding images that
		constitute respective ground truth images for the images
		in the current pipeline.

		This function will search the directory specified by
		:attr:`ground_truth_directory` and will associate each ground truth
		image with the images in the pipeline by file name.

		Therefore, an image titled ``cat321.jpg`` will match with the
		image ``cat321.jpg`` in the :attr:`ground_truth_directory`.
		The function respects each image's label, therefore the image
		named ``cat321.jpg`` with the label ``cat`` will match the image
		``cat321.jpg`` in the subdirectory ``cat`` relative to
		:attr:`ground_truth_directory`.

		Typically used to specify a set of ground truth or gold standard
		images that should be augmented alongside the original images
		of a dataset, such as image masks or semantic segmentation ground
		truth images.

		A number of such data sets are openly available, see for example
		`https://arxiv.org/pdf/1704.06857.pdf <https://arxiv.org/pdf/1704.06857.pdf>`_
		(Garcia-Garcia et al., 2017).

		:param ground_truth_directory: A directory containing the
		 ground truth images that correspond to the images in the
		 current pipeline.
		:type ground_truth_directory: String
		:return: None.
		"""

		num_of_ground_truth_images_added = 0

		# Progress bar
		# progress_bar = tqdm(total=len(self.augmentor_images), desc="Processing", unit=' Images', leave=False)

		if len(self.class_labels) == 1:
			for augmentor_image_idx in range(len(self.augmentor_images)):
				ground_truth_image = os.path.join(ground_truth_directory,
												  self.augmentor_images[augmentor_image_idx].image_file_name.replace(del_str, add_str))
				if os.path.isfile(ground_truth_image):
					self.augmentor_images[augmentor_image_idx].ground_truth = ground_truth_image
					num_of_ground_truth_images_added += 1
					print(self.augmentor_images[augmentor_image_idx].image_file_name.replace(del_str, add_str))

		else:
			for i in range(len(self.class_labels)):
				for augmentor_image_idx in range(len(self.augmentor_images)):
					ground_truth_image = os.path.join(ground_truth_directory,
													  self.augmentor_images[augmentor_image_idx].class_label,
													  self.augmentor_images[augmentor_image_idx].image_file_name.replace(del_str, add_str))
					# print(self.augmentor_images[augmentor_image_idx].image_file_name.replace(del_str, add_str))
					
					if os.path.isfile(ground_truth_image):
						if self.augmentor_images[augmentor_image_idx].class_label == self.class_labels[i][0]:
							# Check files are the same size. There may be a better way to do this.
							original_image_dimensions = \
								Image.open(self.augmentor_images[augmentor_image_idx].image_path).size
							ground_image_dimensions = Image.open(ground_truth_image).size
							if original_image_dimensions == ground_image_dimensions:
								self.augmentor_images[augmentor_image_idx].ground_truth = ground_truth_image
								num_of_ground_truth_images_added += 1
								# progress_bar.update(1)

		# progress_bar.close()

		# May not be required after all, check later.
		if num_of_ground_truth_images_added != 0:
			self.process_ground_truth_images = True

		print("%s ground truth image(s) found." % num_of_ground_truth_images_added)


imageDir = 'data/CVPPP2017/trainA/'
labelDir = 'data/CVPPP2017/trainB/'
p = PipelineGroundtruth(imageDir)
p.ground_truth(labelDir, del_str='rgb', add_str='label')
p.resize(probability=1.0, width=512, height=512, resample_filter='NEAREST')
p.flip_top_bottom(probability=0.5)
p.sample(10)

g = p.keras_generator(batch_size=1)

for s in range(10):
	images, labels = next(g)
	for image, label in zip(images, labels):
		image = image.astype(np.uint8)
		label = label.astype(np.uint8)
		print(image.shape)
		print(label.shape)
		plt.imshow(np.concatenate([image, label], axis=1).astype(np.uint8))
