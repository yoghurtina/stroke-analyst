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

def create_superpixel_image(image_path, n_segments=10000, compactness=100.0):
    image = io.imread(image_path)
    
    # Apply SLIC and create superpixels
    segments = slic(image, n_segments=n_segments, compactness=compactness,channel_axis=None)
    
    # Create an image showing the superpixels overlay
    superpixel_image = label2rgb(segments, image, kind='avg')
    superpixel_image = superpixel_image.astype(bool)
    opened_mask = binary_opening(superpixel_image, square(3))
    
    # Apply morphological closing (close small holes)
    smoothed_mask = binary_closing(opened_mask, square(3))

    return smoothed_mask

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

    return segmented_image



def post_process_mask(binary_mask_path, kernel_size=(5, 5), blur_kernel_size=(1, 1)):
 
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    kernel = np.ones(kernel_size, np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    binary_mask_smoothed = cv2.GaussianBlur(binary_mask, blur_kernel_size, sigmaX=0, sigmaY=0)
    return binary_mask_smoothed

def smooth_mask(binary_mask_path, sigma=1):
    # Load the binary mask
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



import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import nibabel as nib
import cv2
import os

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

