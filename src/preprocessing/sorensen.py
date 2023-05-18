import numpy as np
import segmentation as seg

def sorensen_dice(gt_mask, seg_mask):
    """
    Calculates the Sorensen-Dice coefficient for two binary masks.
    """
    intersection = np.logical_and(gt_mask, seg_mask)
    dice = (2. * intersection.sum()) / (gt_mask.sum() + seg_mask.sum())
    return dice


def dice_coefficient(truth, predicted):
    # Flatten binary masks
    truth = truth.flatten()
    predicted = predicted.flatten()
    
    # Compute true positives, false positives, and false negatives
    true_positives = np.sum(truth * predicted)
    false_positives = np.sum((1 - truth) * predicted)
    false_negatives = np.sum(truth * (1 - predicted))
    
    # Compute Sørensen-Dice coefficient
    coefficient = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
    
    return coefficient
import cv2

# Load ground truth mask and predicted mask
ground_truth = cv2.imread('ground_truth_mask.png', 0)
predicted_mask = cv2.imread('predicted_mask.png', 0)

# Compute Sørensen-Dice coefficient
coefficient = dice_coefficient(ground_truth, predicted_mask)

print('Sørensen-Dice coefficient:', coefficient)

import cv2
from skimage.metrics import binary_dice


# Load ground truth mask
gt_mask = cv2.imread('path/to/ground_truth_mask.png', cv2.IMREAD_GRAYSCALE)

# Load input image
img = cv2.imread('path/to/input_image.png')

# Apply segmentation method
predicted_mask = seg.segmentation(img)

# Calculate Sorensen-Dice coefficient
score = binary_dice(gt_mask, predicted_mask)

print('Sorensen-Dice coefficient:', score)


"""
If you don't have a ground truth mask, then you won't be able to directly calculate the Sørensen-Dice coefficient to measure the accuracy of your image segmentation. However, there are still some ways you could assess the quality of your segmentation without a ground truth mask:

Visual inspection: You could manually inspect the segmented image and compare it to the original image to see if it looks like a reasonable segmentation. You could also compare it to other segmentations produced by different methods, if available.

Comparison to a standard dataset: If your image belongs to a well-known dataset, you could compare your segmentation results to the ground truth provided by the dataset. For example, some popular image segmentation datasets include PASCAL VOC, COCO, and ImageNet.

Domain-specific metrics: Depending on the specific domain of your image, there may be domain-specific metrics that you could use to evaluate the quality of your segmentation. For example, in medical imaging, the Dice similarity coefficient is a common metric used to evaluate the similarity between two volumes.

Keep in mind that without a ground truth mask, your assessment of the quality of the segmentation will be more subjective and less quantitative. It's always best to have a ground truth mask if possible, as it provides a more objective way to evaluate the accuracy of the segmentation.

import os
import cv2
import numpy as np

# Set the paths to the directory containing the images and masks
img_dir = "/path/to/images"
mask_dir = "/path/to/masks"

# Get the list of image files and their corresponding masks
img_files = sorted(os.listdir(img_dir))
mask_files = sorted(os.listdir(mask_dir))

# Loop over the image files and their corresponding masks
for img_file, mask_file in zip(img_files, mask_files):
    # Read the image and mask files
    img = cv2.imread(os.path.join(img_dir, img_file))
    mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)

    # You can now use the image and mask for segmentation and evaluation
    # For example, you can pass the image to your segmentation method and obtain a predicted mask,
    # and then use the Sørensen–Dice coefficient to evaluate the accuracy of the segmentation.

"""