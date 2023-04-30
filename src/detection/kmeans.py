import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

# Load the image data
image = cv2.imread("../data/image.jpg")

# Reshape the image into a 2D array of pixels
pixels = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))

# Perform K-means clustering on the pixel values
kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)

# Assign each pixel to a cluster based on the K-means clustering results
labels = kmeans.labels_

# Reshape the cluster labels back into the original image shape
segmented_image = np.reshape(labels, (image.shape[0], image.shape[1]))

# Apply erosion and dilation to the segmented image to remove small holes and fill gaps
kernel = np.ones((1,1),np.uint8)
segmented_image = cv2.erode(segmented_image.astype(np.uint8), kernel, iterations=1)
segmented_image = cv2.dilate(segmented_image.astype(np.uint8), kernel, iterations=1)

# Create a binary mask by thresholding the segmented image
threshold = 0
mask = np.where(segmented_image > threshold, 255, 0)

# Visualize the segmented mask
plt.imshow(mask, cmap='gray')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage.morphology import binary_erosion, binary_dilation, disk

# Load the image data
image = plt.imread("../data/image.jpg")

# Reshape the image into a 2D array of pixels
pixels = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))

# Perform Gaussian Mixture Model clustering on the pixel values
gmm = GaussianMixture(n_components=3, random_state=0).fit(pixels)

# Assign each pixel to a cluster based on the GMM clustering results
labels = gmm.predict(pixels)

# Reshape the cluster labels back into the original image shape
segmented_image = np.reshape(labels, (image.shape[0], image.shape[1]))

# Create a binary mask of the lesion by selecting the pixels labeled as lesion
lesion_mask = segmented_image == 1

# Perform morphological erosion and dilation operations on the mask to remove small noise
selem = disk(3)
lesion_mask = binary_erosion(lesion_mask, selem)
lesion_mask = binary_dilation(lesion_mask, selem)

# Visualize the segmented image and the lesion mask
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(segmented_image)
ax1.set_title('Segmented Image')
ax2.imshow(lesion_mask)
ax2.set_title('Lesion Mask')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('../data/image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection on the grayscale image
edges = cv2.Canny(gray, 100, 200)

# Reshape the image into a 2D array of pixels
pixels = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))

# Perform K-means clustering on the pixel values
kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)

# Assign each pixel to a cluster based on the K-means clustering results
labels = kmeans.labels_

# Reshape the cluster labels back into the original image shape
segmented_image = np.reshape(labels, (image.shape[0], image.shape[1]))

# Combine the edge map and the segmented image to create a mask
mask = np.logical_and(edges, segmented_image)

# Visualize the mask
plt.imshow(mask, cmap='gray')
plt.show()

