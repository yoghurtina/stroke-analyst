# import cv2
# from matplotlib import pyplot as plt

# # Load the image
# img = cv2.imread('data/good sections/test.jpeg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply thresholding
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# # Plot the original and thresholded images
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(1, 2, 2)
# plt.imshow(thresh, cmap='gray')
# plt.title('Thresholded Image')
# plt.xticks([])
# plt.yticks([])

# plt.show()

# img = io.imread('data/bad sections/1.png')


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology
from skimage import io
# Load image
img = io.imread('data/bad sections/1.png')

# Convert image to grayscale
gray = color.rgb2gray(img)

# Apply median filter to remove noise
median = filters.median(gray)

# Thresholding to create a binary image
binary = median < 0.5

# Morphological opening to remove small objects and smooth edges
opened = morphology.opening(binary, selem=morphology.disk(5))

# Use binary image as a mask to extract background pixels
background = np.zeros_like(img)
background[opened] = img[opened]

# Show the original image and the background
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

ax[0].imshow(img)
ax[1].imshow(background)

plt.show()
