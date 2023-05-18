import cv2
import numpy as np
import os
from image_segmentation import segmentation
import numpy as np
from sklearn.decomposition import PCA
import math

def is_aligned(image, threshold=5):
    # Compute orientation using Hough Transform
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        # Compute the average angle of the detected lines
        angles = [np.arctan2(y2-y1, x2-x1) for x1,y1,x2,y2 in lines[:,0]]
        theta_hough = np.degrees(np.mean(angles))

        # Check if angle is within threshold
        if abs(theta_hough) < threshold:
            return True

    return False

def alignment(image):
    result, mask = segmentation(image)

    if is_aligned(mask):
        return result

    # Apply Hough Transform to detect lines in the image
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Compute angle theta using Hough Transform
    theta_hough = 0
    if lines is not None:
        # Compute the average angle of the detected lines
        angles = [np.arctan2(y2-y1, x2-x1) for x1,y1,x2,y2 in lines[:,0]]
        theta_hough = np.degrees(np.mean(angles))

    # Convert image to grayscale and extract coordinates of nonzero pixels
    coords = np.column_stack(np.where(mask > 0)).astype(np.float32)

    # Compute mean and covariance matrix
    mean, cov = cv2.calcCovarMatrix(coords, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    cov = cov.astype(float)

    # Check if cov has the correct shape
    if cov.shape[0] == 1:
        cov = np.diag(cov[0])

    # Check if cov is positive definite
    if not np.all(np.linalg.eigvals(cov) > 0):
        cov += np.eye(cov.shape[0]) * 1e-5# add small positive constant to diagonal

    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Create an eigenvectors matrix
    if eigenvectors.shape[1] >= 3:
        eigenvector_matrix = np.column_stack((eigenvectors[:, -1], eigenvectors[:, -2], eigenvectors[:, -3]))
    else:
        eigenvector_matrix = np.column_stack((eigenvectors[:, -1], eigenvectors[:, -2]))

    # Determine angle theta using eigenvectors
    theta_eigen = np.degrees(np.arctan2(*eigenvectors[::-1, 0]))

    # If the angle calculated by the Hough Transform is closer to the angle calculated by the eigenvectors,
    # use the Hough angle instead
    if abs(theta_hough - theta_eigen) < 30:
        theta = theta_hough
    else:
        theta = theta_eigen

    # Rotate the image
    (rows, cols) = result.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, theta, 1.0)

    # Compute the new size of the rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_cols = int(rows * sin + cols * cos)
    new_rows = int(rows * cos + cols * sin)

    # Adjust the translation component of the transformation matrix to center the image
    M[0, 2] += (new_cols - cols) / 2
    M[1, 2] += (new_rows - rows) / 2

    rotated = cv2.warpAffine(result, M, (new_cols, new_rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return rotated

path1 = "/home/ioanna/Documents/Thesis/results/preprocessing/segmentation/an"
# path2 = "/home/ioanna/Documents/Thesis/results/preprocessing/segmentation/2"

# Set the path to the folder where you want to save the processed images
output_folder = "/home/ioanna/Documents/Thesis/results/preprocessing/alignment/an"
# output_folder2 = "/home/ioanna/Documents/Thesis/results/preprocessing/alignment"


# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# if not os.path.exists(output_folder2):
#     os.makedirs(output_folder)

#  Loop through all the files in the folder
for filename in os.listdir(path1):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
        # Load the image
        img_path = os.path.join(path1, filename)
        # result, mask = seg.segmentation(img_path)
        rotated = alignment(img_path)
        # print(angle)
        output_img_path = os.path.join(output_folder, "rotated_"+filename)
        cv2.imwrite(output_img_path, rotated)

# for filename in os.listdir(path2):
#     # Check if the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
#         # Load the image
#         img_path = os.path.join(path2, filename)
#         # result, mask = seg.segmentation(img_path)
#         rotated = alignment(img_path)
#         # print(angle)
#         output_img_path = os.path.join(output_folder, "rotated_"+filename)
#         cv2.imwrite(output_img_path, rotated)