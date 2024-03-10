import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def segmentation(image_file):
    img = Image.open(image_file).convert("RGB")
    img = np.array(img)
   
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 20, 200)

    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), -1)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    result = cv2.bitwise_and(img, img, mask=mask)

    return result, mask


def get_segmented(image_file, mask_file):
    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file , cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)

    return segmented_image
