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
    result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR) 

    return result, mask


# # Load the image
# path1 = "/home/ioanna/Documents/Thesis/training/input"

# # Set the path to the folder where you want to save the processed images
# output_folder_path1 = "/home/ioanna/Documents/Thesis/training/segmented"

# # Create the output folder if it doesn't exist
# if not os.path.exists(output_folder_path1):
#     os.makedirs(output_folder_path1)

# # Loop through all the files in the folder
# for filename in os.listdir(path1):
#     # print(filename)
#     # Check if the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
#         # print("his")
#         # Load the image
#         img_path = os.path.join(path1, filename)
#         result, mask = segmentation(img_path)
#         output_img_path = os.path.join(output_folder_path1, "s"+filename)
#         cv2.imwrite(output_img_path, result)

result, mask = segmentation("258allen.png")
cv2.imwrite("fixed2.jpg", result)