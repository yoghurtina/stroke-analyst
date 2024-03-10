import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def segmentation(image_file):
    img = Image.open(image_file).convert("RGB")
    img = np.array(img)
   
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 20, 255)

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
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    # mask = cv2.erode(mask, kernel, iterations=1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR) 

    return result, mask

def rename_images(output_folder_path1):
    files = os.listdir(output_folder_path1)
    count = 1

    for file in files:
        if file.endswith(".png"):
            old_name = os.path.join(output_folder_path1, file)
            new_name = os.path.join(output_folder_path1, str(count) + ".jpg")
            os.rename(old_name, new_name)
            count += 1

def rename_image_masks(output_folder_path1):
    files = os.listdir(output_folder_path1)
    count = 1

    for file in files:
        if file.endswith(".png"):
            old_name = os.path.join(output_folder_path1, file)
            new_name = os.path.join(output_folder_path1, "mask_"+str(count) + ".jpg")
            os.rename(old_name, new_name)
            count += 1

def find_mask(image_file):
    img = Image.open(image_file).convert("RGB")
    img = np.array(img)
   
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 20, 200)

    # Find the contours of the object
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask of the object
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), -1)

    return mask



# Load the image
path1 = "/home/ioanna/Downloads/Ioanna/training-InputAggelos"

# Set the path to the folder where you want to save the processed images
output_folder_path2 = "/home/ioanna/Documents/Thesis/training_data/batch2_aggelos/temp"



if not os.path.exists(output_folder_path2):
    os.makedirs(output_folder_path2)

# Loop through all the files in the folder
for filename in os.listdir(path1):
    
    # print(filename)
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff") or filename.endswith(".JPG"):
        # print("his")
        # Load the image
        img_path = os.path.join(path1, filename)
        result, mask = segmentation(img_path)
        output_mask_path = os.path.join(output_folder_path2, filename)
        cv2.imwrite(output_mask_path, result)

# mask = find_mask("/home/ioanna/Documents/Thesis/results/validation/volumetry/stroke/ground_truth_masks/mask_se_4.jpg")
# cv2.imwrite("/home/ioanna/Documents/Thesis/results/validation/volumetry/stroke/ground_truth_masks/mask_se_4.jpg", mask)

# from PIL import Image
# import numpy as np
# from scipy.ndimage import binary_erosion
# import os

# # Directory paths
# input_directory = '/content/drive/MyDrive/training_data/anastasia/input/'
# output_directory = '/content/drive/MyDrive/training_data/anastasia/segmented/'

# # Create the output directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)

# # Process each image in the input directory
# for filename in os.listdir(input_directory):
#     image_bgr = cv2.imread(os.path.join(input_directory, filename))
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#     sam_result = mask_generator.generate(image_rgb)

#     masks = [
#     mask['segmentation']
#     for mask
#     in sorted(sam_result, key=lambda x: x['area'], reverse=True)
#     ]
#     mask = masks[0]
#     mask = Image.fromarray(mask)
#     mask.save("/content/drive/MyDrive/training_data/anastasia/masks/mask_"+filename)


#     # Load the original image and its mask
#     original_image = Image.open(os.path.join(input_directory, filename))
#     mask = Image.open("/content/drive/MyDrive/training_data/anastasia/masks/mask_"+filename)
#     # Convert the images to numpy arrays
#     original_array = np.array(original_image)
#     mask_array = np.array(mask)
#     print(original_array.shape, mask_array.shape)
#     # Apply the eroded mask to the original image
#     masked_array = original_array.copy()
#     masked_array[mask_array != 255] = 0

#     # Convert the modified array back to an image
#     masked_image = Image.fromarray(masked_array)

#     # Save the segmented image
#     output_path = os.path.join(output_directory, filename)
#     masked_image.save(output_path)

# TODO anastasia segmentation
# TODO manual seg of false
# TODO normalize
# TODO finetune all
# TODO finetune aggelos
# TODO finetune anastasia



