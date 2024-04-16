import cv2
from sklearn.decomposition import PCA
import os
import numpy as np
from module.utils import segmentation_old

def is_aligned(image, threshold=5):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = [np.arctan2(y2-y1, x2-x1) for x1,y1,x2,y2 in lines[:,0]]
        theta_hough = np.degrees(np.mean(angles))

        if abs(theta_hough) < threshold:
            return True

    return False

def alignment(image, mask):
    if is_aligned(mask):
        return image
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    theta_hough = 0
    if lines is not None:
        angles = [np.arctan2(y2-y1, x2-x1) for x1,y1,x2,y2 in lines[:,0]]
        theta_hough = np.degrees(np.mean(angles))

    coords = np.column_stack(np.where(mask > 0)).astype(np.float32)

    mean, cov = cv2.calcCovarMatrix(coords, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    cov = cov.astype(float)

    if cov.shape[0] == 1:
        cov = np.diag(cov[0])

    if not np.all(np.linalg.eigvals(cov) > 0):
        cov += np.eye(cov.shape[0]) * 1e-5# add small positive constant to diagonal

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if eigenvectors.shape[1] >= 3:
        eigenvector_matrix = np.column_stack((eigenvectors[:, -1], eigenvectors[:, -2], eigenvectors[:, -3]))
    else:
        eigenvector_matrix = np.column_stack((eigenvectors[:, -1], eigenvectors[:, -2]))
    theta_eigen = np.degrees(np.arctan2(*eigenvectors[::-1, 0]))

    if abs(theta_hough - theta_eigen) < 30:
        theta = theta_hough
    else:
        theta = theta_eigen

    (rows, cols) = image.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, theta, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_cols = int(rows * sin + cols * cos)
    new_rows = int(rows * cos + cols * sin)

    M[0, 2] += (new_cols - cols) / 2
    M[1, 2] += (new_rows - rows) / 2

    rotated = cv2.warpAffine(image, M, (new_cols, new_rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return rotated
