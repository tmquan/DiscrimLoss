import Augmentor
from Augmentor import *
import random
import numpy as np
import PIL
class PipelinePaired(Pipeline):
	def _execute_with_array_two_images(self, image1, image2):
		"""
		Private method used to execute a pipeline on two arrays.
		:param image1,image2: The images to pass through the pipeline.
		:type image1,image2: Array like object.
		:return: The augmented images.
		"""
		pil_image1 = [PIL.Image.fromarray((image1*255.0).astype('uint8'))]
		pil_image2 = [PIL.Image.fromarray((image2*255.0).astype('uint8'))]
		print(pil_image1)
		print(pil_image2)
		for operation in self.operations:
			r = np.round(random.uniform(0, 1), 1)
			if r <= operation.probability:
				new_seed = random.random()
				random.seed(new_seed)
				pil_image1 = operation.perform_operation(pil_image1)
				random.seed(new_seed)
				pil_image2 = operation.perform_operation(pil_image2)

		# numpy_array1 = np.asarray(pil_image1).astype('float32')/255.0
		# numpy_array2 = np.asarray(pil_image2).astype('float32')/255.0
		numpy_array1 = np.array(pil_image1[0]).astype(np.float32)
		numpy_array2 = np.array(pil_image2[0]).astype(np.float32)

		return numpy_array1,numpy_array2


	def keras_generator_from_array(self, images_X, images_Y, batch_size, image_data_format="channels_last"):
		"""
			Returns an image generator that will sample from the current pipeline
			indefinitely, as long as it is called.
		"""
		if len(images_X) != len(images_Y):
			raise IndexError("The number of images of arrays does not match.")
		while True:
			X = []
			Y = []
			for i in range(batch_size):
				import random
				random_image_index = random.randint(0, len(images_X)-1)
				numpy_array1,numpy_array2 = self._execute_with_array_two_images(images_X[random_image_index,...],
																				images_Y[random_image_index,...])

				w = numpy_array1.shape[0]
				h = numpy_array1.shape[1]

				if np.ndim(numpy_array1) == 2:
					l = 1
				else:
					l = np.shape(numpy_array1)[2]

				if image_data_format == "channels_last":
					numpy_array1 = numpy_array1.reshape(w, h, l)
					numpy_array2 = numpy_array2.reshape(w, h, l)
				elif image_data_format == "channels_first":
					numpy_array1 = numpy_array1.reshape(l, w, h)
					numpy_array2 = numpy_array2.reshape(l, w, h)

				X.append(numpy_array1)
				Y.append(numpy_array2)

			X = np.asarray(X)
			Y = np.asarray(Y)

			yield(X, Y)