# import numpy as np
# import json
# import scipy

# def extract_allen_mask(index):
#     # Load atlas
#     size1 = [528, 320, 456]
#     fid = open('/home/ioanna/Documents/Thesis/src/temp/annotation.raw', 'rb')
#     ANO = np.fromfile(fid, dtype=np.uint32, count=np.prod(size1))
#     fid.close()
#     ANO = np.reshape(ANO, size1)

#     atlas = [None] * ANO.shape[0]
#     for i in range(ANO.shape[0]):
#         atlas[i] = np.squeeze(ANO[i, :, :])

#     slice = atlas[index]
#     return slice

# print(extract_allen_mask(500))

# def json_find(json_data, id_val):
#     # Recursive function to search in a JSON structure for the 'id' value and return the corresponding 'name'
#     if not json_data:
#         return None

#     index = -1
#     for i, item in enumerate(json_data):
#         if item['id'] == id_val:
#             index = i
#             break

#     if index != -1:
#         return json_data[index]['name']

#     for i in range(len(json_data)):
#         ot = json_find(json_data[i]['children'], id_val)
#         if ot is not None:
#             return ot

#     return None

# # # Load JSON data from file
# # with open('acronyms.json') as file:
# #     json_data = json.load(file)

# # # Example usage
# # result = json_find(json_data['msg'], 1111)
# # print(result)

# def region_naming(allen_json, allen_masks, index, lesion_AS, save_dir):
#     # Extract original Allen mask and keep unique
#     ori_allen_labels = extract_allen_mask(523)
#     lesion_AS = np.array(lesion_AS)
#     # Check if the mask exists
#     if index in allen_masks:
#         label_mask = allen_masks[index]
#         print(label_mask)
#         # Sanity check: label mask and lesion_AS mask dimensions must agree
#         if label_mask.shape == lesion_AS.shape:
#             # Filter Allen mask using lesion prediction
#             affected_regions_mask = label_mask * lesion_AS

#             # Acquire interpolated points indices
#             int_indices = np.isin(affected_regions_mask, ori_allen_labels)

#             # Filter out interpolated points, set to zero
#             affected_regions_mask[~int_indices] = 0

#             # Filter duplicates and find affected regions
#             labels = np.unique(np.fix(affected_regions_mask))

#             hit_regions = []
#             for label in labels:
#                 hit_regions.append(json_find(allen_json['msg'], label))

#             with open(save_dir + '/affected_regions.txt', 'w') as filePh:
#                 filePh.write('\n'.join(hit_regions))
            
#             return hit_regions

# region_naming('/home/ioanna/Documents/Thesis/src/temp/acronyms.json', '/home/ioanna/Documents/Thesis/src/temp/allen_masks.mat', "bn2_58.jpg", "sm_8.jpg", "/home/ioanna/Documents/Thesis/src/module")

# # import scipy.io

# # # Load .mat file
# # mat = scipy.io.loadmat('/home/ioanna/Documents/Thesis/src/temp/allen_masks.mat')

# # print(mat.keys())


# # value = mat['None']
# # print(value)

# # # Iterate over variables and print their contents
# # for variable_name in mat:
# #     if variable_name.startswith('__'):
# #         variable_value = mat[variable_name]
# #         print(f'{variable_name}:')
# #         print(variable_value)
# #         print()


import numpy as np
import math
import json

# testing, difficult method
def ccf_to_stereotactic(mm, resolution):
    # CCF size and bregma coordinates in micrometers
    size = np.array([456, 320, 528]) 
    bregma = np.array([216, 18, 228]) 

    # Step 1: Center on bregma
    ccf_coord = np.round((np.array([abs(mm), 0, 0]) + np.array([0, -bregma[1], bregma[2]])) / resolution)
    if mm < 0:
        ccf_coord[0] *= -1
    # Step 2: Rotate the CCF
    ccf_coord[0], ccf_coord[1] = (
        ccf_coord[0] * np.cos(0.0873) - ccf_coord[1] * np.sin(0.0873),
        ccf_coord[0] * np.sin(0.0873) + ccf_coord[1] * np.cos(0.0873)
    )

    # Step 3: Squeeze the DV axis
    ccf_coord[1] *= 0.9434

    # Step 4: Convert to index
    ccf_coord = np.round(ccf_coord[::-1] / resolution).astype(int)

    if np.any(ccf_coord < 0) or np.any(ccf_coord >= size):
        raise ValueError(f"ccf_coord {ccf_coord} is out of bounds for size {size}")
    ccf_index = np.ravel_multi_index(ccf_coord, size)
    return ccf_index

# print(ccf_to_stereotactic(-2.18, 25))

# takes the bregma distance and returns new x coordinate (cartesian)
def find_coordinates(distance):
    distance = -distance # Negative values in distance, get us towards the "head". So, we need to flip the number.
    bregma_base_index = 216
    resolution = 25e-6
    distance_index = distance / resolution
    real_index = bregma_base_index + distance_index

    return math.ceil(real_index)

# This script extract an Allen slice based on the input index.

def extract_allen_slice(index):
    
    size = [528, 320, 456]
    # VOL = 3-D matrix (volume) of atlas Nissl
    with open('atlasVolume.raw', 'rb') as file:
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


def extract_allen_mask(index):
    size = [528, 320, 456]
    # VOL = 3-D matrix (volume) of atlas Nissl
    with open('/home/ioanna/Documents/Thesis/src/temp/annotation.raw', 'rb') as file:
        ANO = np.fromfile(file, dtype=np.uint8, count=np.prod(size))
    ANO = np.reshape(ANO, size)

    # Reshape the numpy array to the desired size
    ANO = ANO.reshape((size[2], size[1], size[0])) # swap the order of dimensions
    ANO = np.transpose(ANO, (2, 1, 0))  # swap dimensions to (456, 320, 528)

    atlas = []
    for i in range(size[0]):
        atlas.append(ANO[i, :, :])
    slice = atlas[index]
    
    return slice

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


def region_naming(json_data, allen_masks, bregma_distance, lesion_AS, save_dir):
    # Extract original Allen mask and keep unique
    ori_allen_labels = extract_allen_mask(find_coordinates(bregma_distance))
    lesion_AS = np.array(lesion_AS)
    
    # if type(bregma_distance) is str:
    #     if index[-4] == 'n':
    #         index = index[:-2] + '.' + index[-2:]
    #         index = float(index[3:])
    #         index *= -1
    #     else:
    #         index = index[:-2] + '.' + index[-2:]
    #         index = float(index[3:])

    # Check if the mask exists
    if bregma_distance in allen_masks:
        label_mask = allen_masks[bregma_distance]
        
        # Sanity check: label mask and lesion_AS mask dimensions must agree
        if label_mask.shape == lesion_AS.shape:
            # Filter Allen mask using lesion prediction
            affected_regions_mask = label_mask * lesion_AS

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
    # else:
    #     print("Mask does not exist")

    return None




# Load JSON data from file and convert to dictionary
with open('/home/ioanna/Documents/Thesis/src/temp/acronyms.json') as file:
    json_data = json.load(file)['msg']
    json_data = {item['id']: item for item in json_data}

import scipy.io
from PIL import Image
# Load .mat file
mat = scipy.io.loadmat('/home/ioanna/Documents/Thesis/src/temp/allen_masks.mat')


# Example usage
lesion_AS = Image.open('sm_8.jpg')

result = region_naming(json_data, mat, 2.2e-3, lesion_AS, "/home/ioanna/Documents/Thesis/src/module")
# print(result)


# Iterate over variables and print their contents
for variable_name in mat:
    if variable_name.startswith('__'):
        variable_value = mat[variable_name]
        print(f'{variable_name}:')
        print(variable_value)
        print()



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

with open('/home/ioanna/Documents/Thesis/src/temp/acronyms.json') as file:
    json_data = json.load(file)

print(json_find(json_data['msg'], 1009))