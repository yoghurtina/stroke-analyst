* PCA for image alignment : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8540825

# Algorithm for Alignment

This code is part of an image alignment algorithm that aims to rotate an input image so that it is aligned with the x-axis. The algorithm consists of several steps, as described below:

The is_aligned function takes an image as input and uses the Hough Transform to detect lines in the image. It then computes the average angle of the detected lines and checks if this angle is within a certain threshold. If the angle is within the threshold, it returns True, indicating that the image is already aligned with the x-axis. Otherwise, it returns False, indicating that the image needs to be rotated.

The align_image function takes an image as input and applies a segmentation algorithm (not shown here) to obtain a binary mask of the image. It then calls the is_aligned function to check if the image is already aligned. If the image is aligned, it returns the input image without any further processing.

If the image is not aligned, the align_image function uses the Hough Transform to detect lines in the binary mask. It then computes the average angle of the detected lines using the same approach as the is_aligned function.

The function then computes the mean and covariance matrix of the non-zero pixel coordinates in the binary mask. If the covariance matrix is not positive definite, a small positive constant is added to the diagonal to make it positive definite.

The eigenvalues and eigenvectors of the covariance matrix are then computed. If there are at least three eigenvectors, the three eigenvectors with the largest eigenvalues are used to create a 3x3 eigenvector matrix. Otherwise, the two largest eigenvectors are used to create a 2x2 matrix.

The angle theta is determined using the eigenvectors. Specifically, the angle between the first eigenvector and the x-axis is calculated.

If the angle calculated by the Hough Transform is within 30 degrees of the angle calculated using the eigenvectors, the Hough angle is used instead of the eigenvector angle.

The input image is rotated by theta degrees using an affine transformation.

If there are at least two eigenvectors, they are drawn on the rotated image.

The rotated image is returned.

Overall, this algorithm uses a combination of the Hough Transform and eigenvectors to align an image with the x-axis. The Hough Transform is used as a quick check to see if the image is already aligned, while the eigenvectors are used to compute a more accurate rotation angle if the image is not aligned.