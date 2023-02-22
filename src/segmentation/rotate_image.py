import cv2
import numpy as np
import os
import image_segmentation as seg
import numpy as np
from sklearn.decomposition import PCA

def align_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the coordinates of the non-zero pixels
    y_coords, x_coords = np.nonzero(gray)

    # Use PCA to find the main axis of the pixel distribution
    X = np.column_stack([x_coords, y_coords])
    
    # Apply PCA to find the main axis of the pixel distribution
    pca = PCA(n_components=2)
    pca.fit(X)
    main_axis = pca.components_[0]

    # Calculate the angle between the main axis and the x-axis
    angle = np.arctan2(main_axis[1], main_axis[0]) * 180 / np.pi

    # Rotate the image to align with the x-axis
    center = ((image.shape[1]-1)/2, (image.shape[0]-1)/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, rotation_matrix,  (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    # Check if the image is reversed
    if np.sum(aligned_image[0, :, :] - aligned_image[-1, :, :]) > 0:
        aligned_image = cv2.rotate(aligned_image, cv2.ROTATE_180)

    # Check if the image is upside down
    if np.sum(aligned_image[:, 0, :] - aligned_image[:, -1, :]) < 0:
        aligned_image = cv2.rotate(aligned_image, cv2.ROTATE_180)

    # Check if the image is flipped horizontally
    if np.mean(aligned_image[0, :, :]) < np.mean(aligned_image[-1, :, :]):
        aligned_image = cv2.flip(aligned_image, 0)

    # Check if the image is flipped vertically
    if np.mean(aligned_image[:, 0, :]) < np.mean(aligned_image[:, -1, :]):
        aligned_image = cv2.flip(aligned_image, 1)

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
        rotated = align_image(result)
        # print(angle)
        output_img_path = os.path.join(output_folder, "rotated_"+filename)
        cv2.imwrite(output_img_path, rotated)