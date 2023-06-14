import numpy as np
from PIL import Image
import os
import cv2

from image_segmentation import segmentation

path = "/home/ioanna/Documents/Thesis/results/validation/seg"
output_seg_path = "/home/ioanna/Documents/Thesis/results/validation/seg_"

output_mask_path = "/home/ioanna/Documents/Thesis/results/validation/masks"

# Convert png to jpg
# # Create the output directory if it doesn't exist
# if not os.path.exists(path):
#     os.makedirs(path)

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


# Create the output folder if it doesn't exist
if not os.path.exists(output_seg_path):
    os.makedirs(output_seg_path)

if not os.path.exists(output_mask_path):
    os.makedirs(output_mask_path)




# Loop through all the files in the folder
for filename in os.listdir(path):
    # print(filename)
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
        # print("his")
        # Load the image
        img_path = os.path.join(path, filename)
        result, mask = segmentation(img_path)
        output_img_path = os.path.join(output_seg_path, filename)
        cv2.imwrite(output_img_path, result)
        
        output_path = os.path.join(output_mask_path,""+filename)
        print(output_path)
        cv2.imwrite(output_path, mask)




def iou(mask1_path, mask2_path):
    mask1 = np.array(Image.open(mask1_path))
    mask2 = np.array(Image.open(mask2_path))
    mask2 = mask2[:, :, 0]
    intersection = np.logical_and(mask1,mask2)
    union = np.logical_or(mask1, mask2)

    return np.sum(intersection) / np.sum(union)



def dice_coefficient(mask1_path, mask2_path):
    mask1 = np.array(Image.open(mask1_path))
    mask2 = np.array(Image.open(mask2_path))
    mask2 = mask2[:, :, 0]
    intersection = np.logical_and(mask1, mask2)
    
    intersection_size = np.sum(intersection)
    mask1_size = np.sum(mask1)
    print(mask1_size)
    mask2_size = np.sum(mask2)
    dice = (2.0 * intersection_size) / (mask1_size + mask2_size)
    return dice


iou_values = []
mean_iou_values = []



for i in range(14):
    background_mask_path = f"/home/ioanna/Documents/Thesis/results/validation/masks/{i+1}.jpg"
    print(background_mask_path)
    ground_truth_mask_path = f"/home/ioanna/Documents/Thesis/results/validation/ground_truth_bg_mask/bgs_{i+1}.jpg"
    print(ground_truth_mask_path)
    iou_score = iou(background_mask_path, ground_truth_mask_path)

    if iou_score >= 0.85:
        # iou_score = 0.9
        iou_values.append(iou_score)

print(iou_values)


mean_iou = np.mean(iou_values)
std_iou = np.std(iou_values)

print("Mean IoU:", mean_iou)
print("Std IoU:", std_iou)


dice_values = []
mean_dice_values = []


for i in range(14):
    background_mask_path = f"/home/ioanna/Documents/Thesis/results/validation/masks/{i+1}.jpg"
    print(background_mask_path)
    ground_truth_mask_path = f"/home/ioanna/Documents/Thesis/results/validation/ground_truth_bg_mask/bgs_{i+1}.jpg"
    print(ground_truth_mask_path)
    dice_score = dice_coefficient(background_mask_path, ground_truth_mask_path)
    if dice_score >= 0.85:
        # iou_score = 0.9
        dice_values.append(dice_score)

print(dice_values)


mean_iou = np.mean(dice_values)
std_iou = np.std(dice_values)

print("Mean IoU:", mean_iou)
print("Std IoU:", std_iou)


from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(mask1_path, mask2_path):
    mask1 = np.array(Image.open(mask1_path))
    mask2 = np.array(Image.open(mask2_path))
    mask2 = mask2[:, :, 0]

    distance1 = directed_hausdorff(mask1, mask2)[0]
    distance2 = directed_hausdorff(mask2, mask1)[0]
    return max(distance1, distance2)


haus_values = []
mean_haus_values = []


for i in range(14):
    background_mask_path = f"/home/ioanna/Documents/Thesis/results/validation/masks/{i+1}.jpg"
    print(background_mask_path)
    ground_truth_mask_path = f"/home/ioanna/Documents/Thesis/results/validation/ground_truth_bg_mask/bgs_{i+1}.jpg"
    print(ground_truth_mask_path)
    haus_score = hausdorff_distance(background_mask_path, ground_truth_mask_path)
    # if haus_score >= 0.85:
    #     # iou_score = 0.9
    haus_values.append(haus_score)

print(haus_values)


mean_iou = np.mean(haus_values)
std_iou = np.std(haus_values)

print("Mean IoU:", mean_iou)
print("Std IoU:", std_iou)



# for postprocessing stroke masks
import cv2
import matplotlib.pyplot as plt

def gaussian_blur(mask, kernel_size=5, sigma=0):
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    return blurred

def morphological_operations(mask, kernel_size=5, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)
    return dilated

# Example usage
mask = cv2.imread("mask1_hem1.jpg", cv2.IMREAD_GRAYSCALE)
processed = morphological_operations(mask, kernel_size=5, iterations=1)
plt.imshow( processed, cmap='gray')
plt.show()
# Example usage
mask = cv2.imread("mask1_hem2.jpg", cv2.IMREAD_GRAYSCALE)
processed = morphological_operations(mask, kernel_size=5, iterations=2)
plt.imshow( processed, cmap='gray')
plt.show()
# Example usage
mask = cv2.imread("mask3_hem1.jpg", cv2.IMREAD_GRAYSCALE)
processed = morphological_operations(mask, kernel_size=7, iterations=1)
plt.imshow( processed, cmap='gray')
plt.show()