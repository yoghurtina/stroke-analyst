import cv2
import numpy as np
import os
import image_segmentation as seg
import numpy as np
from sklearn.decomposition import PCA


def align_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to find edges in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find the lines in the image using the HoughLinesP function
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Check if lines were found
    if lines is not None and len(lines) > 0:
        # Calculate the angle of the first line found
        angle = np.arctan2((lines[0][0][3]-lines[0][0][1]),(lines[0][0][2]-lines[0][0][0])) * 180/np.pi

        # Rotate the image by the calculated angle
        rows,cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        rotated_image = cv2.warpAffine(image,M,(cols,rows))

        return rotated_image
    else:
        # If no lines were found, return the original image
        return image


path1 = "data/results good"

# Set the path to the folder where you want to save the processed images
output_folder = "data/rotated images"


# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#  Loop through all the files in the folder
for filename in os.listdir(path1):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
        # Load the image
        img_path = os.path.join(path1, filename)
        result = seg.segmentation(img_path)
        rotated = align_image(result)
        # print(angle)
        output_img_path = os.path.join(output_folder, "rotated_"+filename)
        cv2.imwrite(output_img_path, rotated)