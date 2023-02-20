import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv2.imread("7.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Invert the thresholded image
thresh = cv2.bitwise_not(thresh)

# Find the contours of the object
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the largest contour
largest_contour = max(contours, key=cv2.contourArea)
mask = np.zeros_like(img)
cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), -1)

# Apply the mask to the original image
result = cv2.bitwise_and(img, mask)

# Show the result
plt.imshow(result[:,:,::-1])
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the image
img = cv2.imread("1.png")

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# apply Canny edge detection
edges = cv2.Canny(blur, 50, 200)

# find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw the contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# plot the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
