import numpy as np
import math
import json
import scipy.io
from PIL import Image

# takes the bregma distance and returns new x coordinate (cartesian)
def find_coordinates(distance):
    distance = -distance # Negative values in distance, get us towards the "head". So, we need to flip the number.
    bregma_base_index = 216
    resolution = 25e-6
    distance_index = distance / resolution
    real_index = bregma_base_index + distance_index

    return math.ceil(real_index)

def extract_allen_slice(index):
    
    size = [528, 320, 456]
    # VOL = 3-D matrix (volume) of atlas Nissl
    with open('raw_data/atlasVolume.raw', 'rb') as file:
        VOL = np.fromfile(file, dtype=np.uint8, count=np.prod(size))
    VOL = np.reshape(VOL, size)

    # Reshape the numpy array to the desired size
    VOL = VOL.reshape((size[2], size[1], size[0])) # swap the order of dimensions
    VOL = np.transpose(VOL, (2, 1, 0))  # swap dimensions to (456, 320, 528)

    atlas = []
    for i in range(size[0]):
        atlas.append(VOL[i, :, :])
    slice = atlas[index]

    return slice

import matplotlib.pyplot as plt
# print(extract_allen_slice(308))
# array = extract_allen_slice(308)
# print(type(array))

# plt.imshow(array, cmap='gray')
# plt.show()

def extract_allen_mask(index):
    size = [528, 320, 456]
    # VOL = 3-D matrix (volume) of atlas Nissl
    with open('/home/ioanna/Documents/Thesis/raw_data/annotation.raw', 'rb') as file:
        ANO = np.fromfile(file, dtype=np.uint32, count=np.prod(size))

    ANO = np.reshape(ANO, size)

    # Reshape the numpy array to the desired size
    ANO = ANO.reshape((size[2], size[1], size[0])) # swap the order of dimensions
    ANO = np.transpose(ANO, (2, 1, 0))  # swap dimensions to (456, 320, 528)

    atlas = []
    for i in range(size[0]):
        atlas.append(ANO[i, :, :])
    slice = atlas[index]
    
    return slice

# import matplotlib.pyplot as plt
# print(extract_allen_mask(308))
# array = extract_allen_mask(308)
# print(type(array))

# plt.imshow(array, cmap='coolwarm')
# plt.show()

def find_region_name(json_data, label):
    # Recursive function to search in a JSON structure for the 'id' value and return the corresponding 'name'
    if not json_data:
        return None

    for item in json_data:
        if item['id'] == label:
            return item['name']

        if 'children' in item:
            region_name = find_region_name(item['children'], label)
            if region_name is not None:
                return region_name

    return None


def json_find(json_data, id_val):
    # Recursive function to search in a JSON structure for the 'id' value and return the corresponding 'name'
    if not json_data:
        return None

    index = -1
    for i, item in enumerate(json_data):
        if item['id'] == id_val:
            index = i
            break

    if index != -1:
        return json_data[index]['name']

    for i in range(len(json_data)):
        ot = json_find(json_data[i]['children'], id_val)
        if ot is not None:
            return ot

    return None
import pandas as pd
import cv2

def region_naming(json_data, bregma_distance, lesion_AS, save_dir):
    # Extract original Allen mask and keep unique
    ori_allen_labels = extract_allen_mask(find_coordinates(bregma_distance))
    ori_allen_labels = np.array(ori_allen_labels)
    # ori_allen_labels = cv2.resize(ori_allen_labels, (600, 600))

    lesion_AS = np.array(lesion_AS)

    # print("allen",ori_allen_labels.shape)
    # print("lesion",lesion_AS.shape)
    # plt.imshow(ori_allen_labels)
    # plt.show()
    
    target_height, target_width = ori_allen_labels.shape
    # print(target_height, target_width)
    # print(target_height, target_width)
    lesion_AS = cv2.resize(lesion_AS, (target_width, target_height))

    # print(lesion_AS.shape)
    # plt.imshow(lesion_AS)
    # plt.show()
    # cv2.imwrite("lesion_AS.png", lesion_AS)
    # print(lesion_AS_resized.shape)
    if ori_allen_labels.shape == lesion_AS.shape:
        
        # Filter Allen mask using lesion prediction
        affected_regions_mask = ori_allen_labels * lesion_AS

        # Acquire interpolated points indices
        int_indices = np.isin(affected_regions_mask, ori_allen_labels)

        # Filter out interpolated points, set to zero
        affected_regions_mask[~int_indices] = 0

        # Filter duplicates and find affected regions
        labels = np.unique(np.fix(affected_regions_mask))

        hit_regions = []
        for label in labels:
            region_name = find_region_name(json_data['msg'], label)
            if region_name is not None:
                hit_regions.append(region_name)

        with open(save_dir + '/affected_regions.txt', 'w') as filePh:
            filePh.write('\n'.join(hit_regions))

        return hit_regions
    else:
        print("Mask dimensions do not match")

    return None

def compare_region_names(ground_truth_regions, predicted_regions):
    common_regions = set(ground_truth_regions) & set(predicted_regions)
    missing_regions = set(ground_truth_regions) - set(predicted_regions)
    extra_regions = set(predicted_regions) - set(ground_truth_regions)
    
    print("Common regions:")
    print(common_regions)
    
    print("Missing regions:")
    print(missing_regions)
    
    print("Extra regions:")
    print(extra_regions)


def calculate_region_iou(ground_truth_regions, predicted_regions):
    intersection = len(set(ground_truth_regions) & set(predicted_regions))
    union = len(set(ground_truth_regions) | set(predicted_regions))
    
    iou = intersection / union
    return iou

# # Load JSON data from file and convert to dictionary
# with open('/home/ioanna/Documents/Thesis/raw_data/acronyms.json') as file:
#     json_data = json.load(file)
#     # json_data = {item['id']: item for item in json_data}
# print(json_data)

# predicted_lesion = Image.open('/home/ioanna/Documents/Thesis/results/validation/volumetry/stroke/computed_masks/17stroke.jpg')
# gt_lesion = Image.open('/home/ioanna/Documents/Thesis/results/validation/volumetry/stroke/ground_truth_masks/mask_se_17.jpg')

# ground_truth_regions = region_naming(json_data,2.22e-3, predicted_lesion, "/home/ioanna/Documents/Thesis/src/module")
# print(ground_truth_regions)
# predicted_regions = region_naming(json_data,2.22e-3, gt_lesion, "/home/ioanna/Documents/Thesis/src/module")
# print(predicted_regions)
# iou = calculate_region_iou(ground_truth_regions, predicted_regions)
# print("IoU:", iou)

# # 0.43 1.82
# # 0.32 0.30
# # 0.22 0.48
# # -0.18 0.39
# # -1.58 0.65
# # 1.92 0.38
# # -0.18 0.47
# # -1.38 0.62
# # -0.38 0.46
# # -0.48 0.43
# # 2.22 0.69



# # lesion_AS = Image.open('/src/module/mask3_hem1.jpg')
# # allen_masks = "/raw_data/allen_masks/-2.7.csv"
# # 
# # result = region_naming(json_data, allen_masks,-2.68e-3, lesion_AS, "/home/ioanna/Documents/Thesis/src/module")
# # print(result)

