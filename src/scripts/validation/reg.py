import SimpleITK as sitk
import numpy as np
from PIL import Image
import cv2

from skimage.metrics import structural_similarity as compare_ssim
from skimage.measure import label, regionprops


from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff


def calculate_mutual_information(image1, image2, num_bins=256):
    # Calculate the joint histogram
    ref_img = image1[:, :, 0]
    reg_img = image2[:image1.shape[0], :image1.shape[1]]

    # if image1.shape != image2.shape:
    #     # Resize one of the images to match the size of the other image
    #     image1 = cv2.resize(image1, image2.shape[::-1])
    joint_histogram, _, _ = np.histogram2d(
        ref_img.ravel(), reg_img.ravel(), bins=num_bins)

    # Normalize the joint histogram
    joint_histogram = joint_histogram / np.sum(joint_histogram)

    # Calculate marginal histograms
    hist1, _ = np.histogram(ref_img.ravel(), bins=num_bins)
    hist2, _ = np.histogram(reg_img.ravel(), bins=num_bins)

    # Normalize the marginal histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # Calculate the mutual information
    epsilon = np.finfo(float).eps  # small value to avoid log(0)
    mi = np.sum(joint_histogram * np.log2((joint_histogram + epsilon) / (hist1[:, None] * hist2[None, :])))
    
    return mi



ref_image_path = '/home/ioanna/Documents/Thesis/temp/1/mapped_allen_section.jpg'
registered_image_path1 = '/home/ioanna/Documents/Thesis/temp/1/rigid.jpg'
registered_image_path2 = '/home/ioanna/Documents/Thesis/temp/1/affine.jpg'
registered_image_path3 = '/home/ioanna/Documents/Thesis/temp/1/non_rigid.jpg'


# initial moving image
orig = Image.open(ref_image_path)
# moving image array
orig_array = np.array(orig)


# registred image
reg1 = Image.open(registered_image_path1)
# convert registered to array
reg_array1 = np.array(reg1)

# registred image
reg2 = Image.open(registered_image_path2)
# convert registered to array
reg_array2 = np.array(reg2)

# registred image
reg3 = Image.open(registered_image_path3)
# convert registered to array
reg_array3 = np.array(reg3)


def normalize_image(img):
    """
    Normalize an image to have zero mean and unit variance.

    """
    
    return (img - np.mean(img)) / np.std(img)

def calculate_ncc(ref_img, reg_img):
    """
    Calculate the NCC metric between two images.
    """
    ref_img_norm = normalize_image(ref_img)
    print(ref_img_norm.shape)
    reg_img_norm = normalize_image(reg_img[:ref_img.shape[0], :ref_img.shape[1]])
    print(reg_img_norm.shape)
    ref_img_norm = ref_img_norm[:, :, 0]
    print(ref_img_norm.shape)
    
    ncc = np.sum((ref_img_norm - np.mean(ref_img_norm)) * (reg_img_norm - np.mean(reg_img_norm))) / \
        (np.std(ref_img_norm) * np.std(reg_img_norm) * ref_img_norm.size)
    return ncc

def calculate_ssim(image1, image2):
    """Calculate Structural Similarity Index (SSIM) between two images."""
    image1 = image1[:, :, 0]
    image2 = image2[:image1.shape[0], :image1.shape[1]]


    ssim = compare_ssim(image1, image2, data_range=image2.max() - image2.min())
    return ssim


def calculate_hd(ref_img, reg_img):
    """
    Calculate the HD metric between two images.
    """
    ref_img = ref_img[:, :, 0]
    reg_img = reg_img[:ref_img.shape[0], :ref_img.shape[1]]

    hd = max(directed_hausdorff(ref_img, reg_img)[0], directed_hausdorff(reg_img, ref_img)[0])
    return hd

def calculate_ji(ref_img, reg_img):
    """
    Calculate the JI metric between two images.
    """
    ref_img = ref_img[:, :, 0]
    reg_img = reg_img[:ref_img.shape[0], :ref_img.shape[1]]


    ji = np.sum(np.logical_and(ref_img > 0, reg_img > 0)) / np.sum(np.logical_or(ref_img > 0, reg_img > 0))
    return ji


def calculate_mse(image1, image2):
    """Calculate Mean Squared Error (MSE) between two images."""
    image1 = image1[:, :, 0]
    image2 = image2[:image1.shape[0], :image1.shape[1]]

    squared_diff = (image1.astype(np.float64) - image2.astype(np.float64)) ** 2
    mse = np.mean(squared_diff)
    return mse


def calculate_psnr(image1, image2, max_value=255):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr



n1 = calculate_ncc(orig_array, reg_array1)
n2 =calculate_ncc(orig_array, reg_array2)
n3 =calculate_ncc(orig_array, reg_array3)

print('NCC', n1, n2, n3)

h1 =calculate_hd(orig_array, reg_array1)
h2 =calculate_hd(orig_array, reg_array2)
h3 =calculate_hd(orig_array, reg_array3)

print('HD', h1, h2, h3)

j1 =calculate_ji(orig_array, reg_array1)
j2 =calculate_ji(orig_array, reg_array2)
j3 =calculate_ji(orig_array, reg_array3)

print('JI', j1, j2, j3)

m1 =calculate_mutual_information(orig_array, reg_array1)
m2 =calculate_mutual_information(orig_array, reg_array2)
m3 =calculate_mutual_information(orig_array, reg_array3)

print('MI', m1, m2, m3)

s1 =calculate_ssim(orig_array, reg_array1)
s2 =calculate_ssim(orig_array, reg_array2)
s3 =calculate_ssim(orig_array, reg_array3)

print('SSIM', s1, s2, s3)

p1 =calculate_psnr(orig_array, reg_array1)
p2 =calculate_psnr(orig_array, reg_array2)
p3 =calculate_psnr(orig_array, reg_array3)

print('PSNR', p1, p2, p3)

# mse
mse1 = calculate_mse(orig_array, reg_array1)
mse2 = calculate_mse(orig_array, reg_array2)
mse3 = calculate_mse(orig_array, reg_array3)

print('MSE', mse1, mse2, mse3)