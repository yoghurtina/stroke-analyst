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


    # with Image.open(image_file) as img:
    #     img = img.convert('L')
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
    uniform_norm = uniform_norm.astype('float')

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

# img = equalize_this(image_file='test2.jpg', with_plot=True)
# path = "/home/ioanna/Documents/Thesis/training/segmented"
# path1 = "/home/ioanna/Documents/Thesis/training/stroke_extracted"
# path2 = "/home/ioanna/Documents/Thesis/raw_data/coordinated_to_allen"


# output_folder_path = "/home/ioanna/Documents/Thesis/training/normalized"
# output_folder_path1 = "/home/ioanna/Documents/Thesis/training/stroke_extracted_normalized"
# output_folder_path2 = "/home/ioanna/Documents/Thesis/results/preprocessing/normalization"

# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

# if not os.path.exists(output_folder_path1):
#     os.makedirs(output_folder_path1)

# if not os.path.exists(output_folder_path2):
#     os.makedirs(output_folder_path2)

# for filename in os.listdir(path):
#     # Check if the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
#         # Load the image
#         img_path = os.path.join(path, filename)
#         result = equalize_this(img_path, with_plot=False, gray_scale=True)
#         output_img_path = os.path.join(output_folder_path, filename)
#         cv2.imwrite(output_img_path, result)


# for filename in os.listdir(path2):
#     # Check if the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
#         # Load the image
#         print(filename)
#         img_path = os.path.join(path2, filename)
#         result = equalize_this(img_path, with_plot=False, gray_scale=True)
#         output_img_path = os.path.join(output_folder_path2, filename)
#         cv2.imwrite(output_img_path, result)

path1 = "/home/ioanna/Documents/Thesis/raw_data/atlas_seg"
output_folder_path1 = "/home/ioanna/Documents/Thesis/raw_data/atlas_norm"


for filename in os.listdir(path1):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
        # Load the image
        img_path = os.path.join(path1, filename)
        result = equalize_this(img_path, with_plot=False, gray_scale=True)
        output_img_path = os.path.join(output_folder_path1, filename)
        cv2.imwrite(output_img_path, result)


import os

def rename_images(output_folder_path1):
    files = os.listdir(output_folder_path1)
    count = 1

    for file in files:
        if file.endswith(".png"):
            old_name = os.path.join(output_folder_path1, file)
            new_name = os.path.join(output_folder_path1, str(count) + ".jpg")
            os.rename(old_name, new_name)
            count += 1

# Provide the directory path where your images are located
directory_path = "/home/ioanna/Documents/Thesis/raw_data/atlas_norm"


rename_images(directory_path)
