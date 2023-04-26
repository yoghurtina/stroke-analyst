import SimpleITK as sitk
import numpy as np
from scipy import stats

def validate_registration(orig_img, reg_img):
    """
    Perform statistical validation of registration results using the mean and
    standard deviation of the intensity values in the images.

    Parameters:
    orig_img (ndarray): The original image.
    reg_img (ndarray): The registered image.

    Returns:
    result (dict): A dictionary containing the results of the validation.
    """

    # Compute the mean and standard deviation of the intensity values in the original image
    orig_mean = np.mean(orig_img)
    orig_std = np.std(orig_img)

    # Compute the mean and standard deviation of the intensity values in the registered image
    reg_mean = np.mean(reg_img)
    reg_std = np.std(reg_img)

    # Compute the differences between the mean and standard deviation of the original and registered images
    mean_diff = np.abs(reg_mean - orig_mean)
    std_diff = np.abs(reg_std - orig_std)

    # Compute the p-values for the differences using a two-tailed t-test
    mean_pval = stats.ttest_ind(orig_img.flatten(), reg_img.flatten(), equal_var=False)[1]
    std_pval = stats.ttest_ind(orig_img.flatten(), reg_img.flatten(), equal_var=False)[1]

    # Store the results in a dictionary
    result = {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'mean_pval': mean_pval,
        'std_pval': std_pval
    }

    return result

import numpy as np

def mad(img1, img2):
    return np.mean(np.abs(img1 - img2))

def rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    return rmse

import numpy as np
from scipy.stats import pearsonr

def pcc(img1, img2):
    # Flatten the images into 1D arrays
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    # Calculate the PCC between the two arrays
    pcc, _ = pearsonr(img1_flat, img2_flat)
    return pcc

# the images should be binary masks of the segmented regions. The function calculates the intersection between the two masks, computes the union of the masks, 
# and then returns the IoU score.
# The IoU score ranges from 0 to 1, where 1 indicates a perfect overlap between the segmented regions and 0 indicates no overlap
def iou(img1, img2):
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


import numpy as np
from scipy.spatial.distance import directed_hausdorff

# binary masks of original ang registered
def hausdorff(img1, img2):
    # Get the coordinates of the points in the segmented regions
    points1 = np.array(np.nonzero(img1)).T
    points2 = np.array(np.nonzero(img2)).T
    
    # Calculate the directed Hausdorff distance between the two sets of points
    dist1 = directed_hausdorff(points1, points2)[0]
    dist2 = directed_hausdorff(points2, points1)[0]
    
    # Return the maximum distance between the two sets of points
    hd_score = max(dist1, dist2)
    return hd_score


moving_image_path = ''
registered_image_path = ''

# initial moving image
orig = sitk.ReadImage(moving_image_path)
# moving image array
orig_array = sitk.GetArrayFromImage(orig)


# registred image
reg = sitk.ReadImage(registered_image_path)
# convert registered to array
reg_array = sitk.GetArrayFromImage(reg)

result = validate_registration(orig_array, reg_array)
print(result)

result = mad(orig_array, reg_array)
print(result)

result = rmse(orig_array, reg_array)
print(result)

result = pcc(orig_array, reg_array)
print(result)

result = iou(orig_array, reg_array)
print(result)


result = hausdorff(orig_array, reg_array)
print(result)
