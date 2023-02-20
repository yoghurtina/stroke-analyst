import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# img = cv2.imread("CA1R/7.jpg")
img = cv2.imread("diff.JPG")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 20, 200)

# Apply dilation to thicken the edges
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(edges, kernel, iterations=1)

# Find the contours of the object
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask of the object
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), -1)

# Apply the mask to the original image
result = cv2.bitwise_and(img, img, mask=mask)

# Show the result
plt.imshow(result[:,:,::-1])
plt.show()
