# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def open_image(path):
    image = Image.open(path)
    image = np.asarray(image)
    return image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989, 0.5870, 0.1140])

path = "1.png"

def superpixels(path):
	img = open_image(path)

	# load the image and convert it to a floating point data type
	image = img_as_float(open_image(path))

	# loop over the number of segments
	for numSegments in (100, 200, 500):
		# apply SLIC and extract (approximately) the supplied number
		# of segments
		segments = slic(image, n_segments = numSegments, sigma = 5)
		# show the output of SLIC
		fig = plt.figure("Superpixels -- %d segments" % (numSegments))
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(mark_boundaries(image, segments))
		plt.axis("off")

	# show the plots
	plt.show()

superpixels(path)