import numpy as np
from PIL import Image
import os
import cv2
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt




# for postprocessing stroke masks

def gaussian_blur(mask, kernel_size=5, sigma=0):
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    return blurred

def morphological_operations(mask, kernel_size=5, iterations=1):
    mask = np.array(Image.open(mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)
    return dilated


def hausdorff_distance(mask1_path, mask2_path):
    mask1 = np.array(Image.open(mask1_path))
    mask2 = np.array(Image.open(mask2_path))
    # mask2 = mask2[:, :, 0]
    mask2=cv2.resize(mask2, mask1.shape[::-1])

    distance1 = directed_hausdorff(mask1, mask2)[0]
    distance2 = directed_hausdorff(mask2, mask1)[0]
    return max(distance1, distance2)



def iou(mask1_path, mask2_path):
    mask1 = np.array(Image.open(mask1_path))
    mask2 = np.array(Image.open(mask2_path))
    # mask2 = mask2[:, :, 0]
    mask2=cv2.resize(mask2, mask1.shape[::-1])
    intersection = np.logical_and(mask1,mask2)
    union = np.logical_or(mask1, mask2)

    return np.sum(intersection) / np.sum(union)



def dice_coefficient(mask1_path, mask2_path):
    mask1 = np.array(Image.open(mask1_path))
    mask2 = np.array(Image.open(mask2_path))
    # mask2 = mask2[:, :, 0]
    mask2=cv2.resize(mask2, mask1.shape[::-1])

    intersection = np.logical_and(mask1, mask2)
    
    intersection_size = np.sum(intersection)
    mask1_size = np.sum(mask1)
    print(mask1_size)
    mask2_size = np.sum(mask2)
    dice = (2.0 * intersection_size) / (mask1_size + mask2_size)
    return dice



# Convert png to jpg. write a function for that

# # Loop over the files in the input directory
# for filename in os.listdir(path):
#     # Check if the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # Open the image and convert it to JPEG format
#         image_path = os.path.join(path, filename)
#         image = Image.open(image_path)
#         image = image.convert("RGB")
#         # Save the image to the output directory with a .jpg extension
#         output_path = os.path.join(path, os.path.splitext(filename)[0] + ".jpg")
#         image.save(output_path)

iou_values = []
dice_values = []
haus_values = []

for i in range(4, 19):
    background_mask_path = f"/home/ioanna/Documents/Thesis/results/validation/volumetry/stroke/computed_masks/{i+1}stroke.jpg"
    ground_truth_mask_path = f"/home/ioanna/Documents/Thesis/results/validation/volumetry/stroke/ground_truth_masks/mask_se_{i+1}.jpg"

    iou_score = iou(background_mask_path, ground_truth_mask_path)
    # if iou_score >= 0.85:
    iou_values.append(iou_score)

    dice_score = dice_coefficient(background_mask_path, ground_truth_mask_path)
    # if dice_score >= 0.85:
    dice_values.append(dice_score)

    haus_score = hausdorff_distance(background_mask_path, ground_truth_mask_path)
    haus_values.append(haus_score)
    

print(iou_values)
print(dice_values)
print(haus_values)

mean_iou = np.mean(iou_values)
std_iou = np.std(iou_values)
print("Mean IoU:", mean_iou)
print("Std IoU:", std_iou)

mean_dice = np.mean(dice_values)
std_dice = np.std(dice_values)
print("Mean Dice:", mean_dice)
print("Std Dice:", std_dice)

mean_haus = np.mean(haus_values)
std_haus = np.std(haus_values)
print("Mean Hausdorff:", mean_haus)
print("Std Hausdorff:", std_haus)

# path1 = "/home/ioanna/Documents/Thesis/results/validation/volumetry/stroke/computed_masks"
# for filename in os.listdir(path1): 
#     # print(filename)
#     # Check if the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
#         # print("his")
#         # Load the image
#         img_path = os.path.join(path1, filename)
#         mask = morphological_operations(img_path, kernel_size=5, iterations=1)
#         output_mask_path = os.path.join("home/ioanna/Documents/Thesis/results/validation/volumetry/stroke/computed_masks", filename)
#         cv2.imwrite(output_mask_path, mask)
