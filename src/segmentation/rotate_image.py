import cv2
import numpy as np
import os
import image_segmentation as seg
import numpy as np
from sklearn.decomposition import PCA


def rotate_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reshape the image into a 1D array
    X = gray.reshape(-1, 1)

    # Check if the image is flat
    if X.shape[0] == 1 or X.shape[1] == 1:
        return image

    # Apply PCA to find the main axis of the pixel distribution
    pca = PCA(n_components=2)
    pca.fit(X)
    main_axis = pca.components_[0]

    # Calculate the angle between the main axis and the x-axis
    angle = np.arctan2(main_axis[1], main_axis[0]) * 180 / np.pi

    # Rotate the image to align with the x-axis
    center = tuple(np.array(gray.shape) // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, rotation_matrix, gray.shape, flags=cv2.INTER_LINEAR)

    return aligned_image

        

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
        rotate_image(result)
        output_img_path = os.path.join(output_folder, "rotated_"+filename)
        cv2.imwrite(output_img_path, result)