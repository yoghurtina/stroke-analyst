import cv2
import numpy as np
import random

# Load the two images
image1 = cv2.imread("fixed.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("moving.jpg", cv2.IMREAD_GRAYSCALE)

# # Create a feattialize the SIFT detector
# sift = cv2.SIFT_create()

# # Detect keypoints and compute descriptors for both images
# keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# # Create a BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# # Match descriptors
# matches = bf.match(descriptors1, descriptors2)

# # Sort the matches by distance
# matches = sorted(matches, key=lambda x: x.distance)

# # Select the top N matches
# N = 10
# matches = matches[:N]

# # Draw the matched keypoints on the images
# matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Display the matched image
# cv2.imshow('Matched Keypoints', matched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
detector1 = cv2.SIFT_create()
detector2 = cv2.ORB_create()

# Detect landmarks in both images using each detector
keypoints1_1 = detector1.detect(image1)
keypoints1_2 = detector2.detect(image1)

keypoints2_1 = detector1.detect(image2)
keypoints2_2 = detector2.detect(image2)

# Perform majority voting to combine the landmark detections in image1
landmarks1 = []
for kp1, kp2 in zip(keypoints1_1, keypoints1_2):
    if kp1.pt == kp2.pt:
        landmarks1.append(kp1.pt)

# Perform majority voting to combine the landmark detections in image2
landmarks2 = []
for kp1, kp2 in zip(keypoints2_1, keypoints2_2):
    if kp1.pt == kp2.pt:
        landmarks2.append(kp1.pt)

# Draw the landmarks on the images
image1_with_landmarks = cv2.drawKeypoints(image1, landmarks1, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_with_landmarks = cv2.drawKeypoints(image2, landmarks2, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the images with landmarks
cv2.imshow('Image 1 with Landmarks', image1_with_landmarks)
cv2.imshow('Image 2 with Landmarks', image2_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()

