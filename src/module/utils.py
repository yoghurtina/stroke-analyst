import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import os
from skimage import io
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage import io
from skimage.morphology import binary_opening, binary_closing, square
import os, shutil
import streamlit as st
import matplotlib.pyplot as plt

def create_superpixel_image(image_path, n_segments=10000, compactness=100.0):
    image = io.imread(image_path)
    segments = slic(image, n_segments=n_segments, compactness=compactness,channel_axis=None)
    superpixel_image = label2rgb(segments, image, kind='avg')
    superpixel_image = superpixel_image.astype(bool)
    opened_mask = binary_opening(superpixel_image, square(3))
    smoothed_mask = binary_closing(opened_mask, square(3))

    return smoothed_mask

def create_binary_mask(image_path, mask_image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(mask_image_path, clean_mask)
    return mask_image_path

def segmentation_old(image_file):
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
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    return segmented_image

def get_segmented_hemispheres(image_file, mask_file):
    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file , cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    return segmented_image

def post_process_mask(binary_mask_path, kernel_size=(4, 4), blur_kernel_size=(1, 1)):
 
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones(kernel_size, np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    binary_mask_smoothed = cv2.GaussianBlur(binary_mask, blur_kernel_size, sigmaX=0, sigmaY=0)
    return binary_mask_smoothed

def smooth_mask(binary_mask_path, sigma=1):
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    binary_mask_smoothed = gaussian_filter(binary_mask, sigma=sigma)
    return binary_mask_smoothed

def save_array_as_image(array, file_path):
    if array.dtype != np.uint8:
        array = (255 * (array - array.min()) / (array.max() - array.min())).astype(np.uint8)
    cv2.imwrite(file_path, array)


def save_uploadedfile(uploadedfile, path):
    with open(os.path.join(path),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved uploaded image to a temporary folder")

def delete_foldercontents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def convert_to_grayscale(input_path):
    with Image.open(input_path) as img:
        img = img.convert('L')
    return img

def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

    return image_src
    

def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq

def equalize_this(image_file, with_plot=False, gray_scale=True, bins=256):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    if not gray_scale:
        r_image = image_src[:, :, 0]
        g_image = image_src[:, :, 1]
        b_image = image_src[:, :, 2]

        r_image_eq = enhance_contrast(image_matrix=r_image)
        g_image_eq = enhance_contrast(image_matrix=g_image)
        b_image_eq = enhance_contrast(image_matrix=b_image)

        image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
        cmap_val = None
    else:
        image_eq = enhance_contrast(image_matrix=image_src)
        cmap_val = 'gray'

    if with_plot:
        fig = plt.figure(figsize=(10, 20))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Equalized")

        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_eq, cmap=cmap_val)
        plt.show()
        return True
    return image_eq


def save_uploadedfile(uploadedfile, path):
    with open(os.path.join(path),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved uploaded image to a temporary folder")

def delete_foldercontents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def create_mask_from_path(path, img_shape, scaling_factor):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    prev_x = prev_y = None

    for segment in path:
        command = segment[0]
        # Apply scaling factor and convert points to integers
        points = [int(float(p) / scaling_factor) for p in segment[1:]]

        if command == 'M':
            # 'M' sets the starting point for new sub-paths
            if len(points) >= 2:
                prev_x, prev_y = points[0], points[1]
        elif command == 'L':
            # 'L' draws a line from the current point to the given point
            if len(points) >= 2 and prev_x is not None and prev_y is not None:
                cv2.line(mask, (prev_x, prev_y), (points[0], points[1]), 255, thickness=1)
                prev_x, prev_y = points[0], points[1]
        elif command == 'Q':
            # 'Q' draws a quadratic Bézier curve from the current point to the end point with a single control point
            if len(points) >= 4 and prev_x is not None and prev_y is not None:
                # For simplicity, approximating quadratic Bézier curve by a line from current point to end point
                cv2.line(mask, (prev_x, prev_y), (points[2], points[3]), 255, thickness=1)
                prev_x, prev_y = points[2], points[3]

    return mask

def split_image(img, mask):
    fill_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(mask, fill_mask, (0, 0), 255)  # Start flood fill from top-left corner
    left_part = cv2.bitwise_and(img, img, mask=mask)
    right_part = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    return left_part, right_part

def resize_image_aspect_ratio(image_path, output_path, max_size=400):
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    scaling_factor = max_size / max(original_width, original_height)
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    resized_image.save(output_path)
    print(f"Image has been resized to {new_width}x{new_height} and saved successfully!")
    
    return resized_image, scaling_factor# Example usage

def translate_bbox_to_original(bbox_resized, scaling_factor):
    """
    Translate bounding box coordinates from the resized dimensions back to the original image dimensions.

    Parameters:
    - bbox_resized: The bounding box in the resized image, as a dictionary with keys 'x', 'y', 'width', 'height'.
    - scaling_factor: The scaling factor used to resize the image.

    Returns:
    - A dictionary containing the bounding box coordinates translated back to the original image dimensions.
    """
    
    # Translate the bounding box coordinates
    bbox_original = {
        'x': bbox_resized['x'] / scaling_factor,
        'y': bbox_resized['y'] / scaling_factor,
        'width': bbox_resized['width'] / scaling_factor,
        'height': bbox_resized['height'] / scaling_factor
    }
    return bbox_original
