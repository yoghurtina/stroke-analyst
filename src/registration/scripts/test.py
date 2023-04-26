import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def resample_contour(contour, num_points):
    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)

    # Calculate the step size to achieve the desired number of points
    step_size = perimeter / num_points

    # Initialize variables for resampling
    current_distance = 0
    resampled_contour = []
    prev_point = contour[0][0]

    # Iterate through the contour points and resample
    for point in contour[1:]:
        # Calculate the distance between the current point and the previous point
        distance = np.linalg.norm(point - prev_point)

        # Add the current point to the resampled contour
        if current_distance + distance >= step_size:
            # Interpolate between the previous point and the current point to find the resampled point
            alpha = (step_size - current_distance) / distance
            resampled_point = prev_point + alpha * (point - prev_point)
            resampled_contour.append(resampled_point)

            # Reset the current distance
            current_distance = 0
        else:
            # Increment the current distance
            current_distance += distance

        # Update the previous point
        prev_point = point

    # Convert the resampled contour to a NumPy array
    resampled_contour = np.array(resampled_contour, dtype=np.int32)

    return resampled_contour


def obtain_fixed_contour_points(image_file):
    img = Image.open(image_file)
    img = np.array(img)
   
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR) 

        # Extract contour points and convert to 2D array
    contour_points = largest_contour.reshape(-1, 2)

    # Store contour coordinates in a NumPy array
    contour_coords = np.array(contour_points)

    # Save contour coordinates to a CSV file
    with open('fixedPointSet.pts', 'w') as f:
        f.write('index\n')
        f.write(f'{len(contour_coords)}')
        for i, (x, y) in enumerate(contour_coords):
            f.write(f'\n{x} {y}')
    # Plot the largest contour
    plt.imshow(mask, cmap='gray')
    plt.plot(contour_points[:, 0], contour_points[:, 1], 'r', linewidth=2)
    plt.show()
    return len(contour_coords)

img = 'fixed.jpg'
# fixed_num=obtain_fixed_contour_points(img)


def obtain_moving_contour_points(image_file, num_points):
    img = Image.open(image_file)
    img = np.array(img)
   
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 20, 200)

    # Apply dilation to thicken the edges
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)

    # Find the contours of the object
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    ## Calculate the perimeter of the contour
    perimeter = cv2.arcLength(largest_contour, True)
    print(perimeter)

    # Calculate the step size to achieve the desired number of points
    step_size = perimeter/ num_points
    print(step_size)
    # Initialize variables for resampling
    current_distance = 0
    resampled_contour = []
    prev_point = largest_contour[0][0]

    # Iterate through the contour points and resample
    for point in largest_contour[1:]:
        # Calculate the distance between the current point and the previous point
        distance = np.linalg.norm(point - prev_point)

        # Add the current point to the resampled contour
        if current_distance + distance >= step_size:
            # Interpolate between the previous point and the current point to find the resampled point
            alpha = (step_size - current_distance) / distance
            resampled_point = prev_point + alpha * (point - prev_point)
            resampled_contour.append(resampled_point)

            # Reset the current distance
            current_distance = 0
        else:
            # Increment the current distance
            current_distance += distance

        # Update the previous point
        prev_point = point

    # Convert the resampled contour to a NumPy array
    resampled_contour = np.array(resampled_contour, dtype=np.int32)

    # Store contour coordinates in a NumPy array
    print(len(resampled_contour))

    # # Save contour coordinates to a CSV file
    # with open('movingPointSet.pts', 'w') as f:
    #     f.write('index\n')
    #     f.write(f'{len(resampled_contour)}')
    #     for i, (x, y) in enumerate(resampled_contour):
    #         f.write(f'\n{x} {y}')


    # Create a mask of the object
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [resampled_contour], 0, (255, 255, 255), -1)

    # Plot the resampled contour
    plt.imshow(mask, cmap='gray')
    plt.plot(resampled_contour[:, 0, 0], resampled_contour[:, 0, 1], 'r', linewidth=2)
    plt.show()

    return mask, resampled_contour

obtain_moving_contour_points('moving.jpg', 766)