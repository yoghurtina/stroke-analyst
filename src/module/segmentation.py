import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def segmentation(image_file):
    img = Image.open(image_file).convert("RGB")
    img = np.array(img)
   
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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

    # Apply erosion to remove noise in outer edges
    kernel = np.ones((5,5),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)
    # result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR) 

    return result, mask
