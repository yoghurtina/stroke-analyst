import numpy as np
import json

def extract_allen_mask(index):
    # Load atlas
    size1 = [528, 320, 456]
    fid = open('annotation.raw', 'rb')
    ANO = np.fromfile(fid, dtype=np.uint32, count=np.prod(size1))
    fid.close()
    ANO = np.reshape(ANO, size1)

    atlas = [None] * ANO.shape[0]
    for i in range(ANO.shape[0]):
        atlas[i] = np.squeeze(ANO[i, :, :])

    slice = atlas[index]
    return slice

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

# # Load JSON data from file
# with open('acronyms.json') as file:
#     json_data = json.load(file)

# # Example usage
# result = json_find(json_data['msg'], 1111)
# print(result)

def region_naming(allen_json, allen_masks, index, lesion_AS, save_dir):
    # Extract original allen mask and keep unique
    ori_allen_labels = np.unique(extract_allen_mask(1))

# Cast index to number if it's a string
    if type(index) is str:
        if index[-4] == 'n':
            index = index[:-2] + '.' + index[-2:]
            index = float(index[3:])
            index *= -1
        else:
            index = index[:-2] + '.' + index[-2:]
            index = float(index[3:])

    # Check if the mask exists
    if index in allen_masks:
        label_mask = allen_masks[index]

        # Sanity check: label mask and lesion_AS mask dimensions must agree
        if label_mask.shape == lesion_AS.shape:
            # Filter allen mask using lesion prediction
            affected_regions_mask = label_mask * lesion_AS

            # Acquire interpolated points indices
            int_indices = np.isin(affected_regions_mask, ori_allen_labels)

            # Filter out interpolated points and set them to zero
            affected_regions_mask[~int_indices] = 0

            # Filter duplicates and find affected regions
            labels = np.unique(affected_regions_mask.astype(int).flatten())
            hit_regions = []
            for label in labels:
                hit_regions.append(json_find(allen_json['msg'], label))

            with open(save_dir + '/affected_regions.txt', 'w') as filePh:
                filePh.write('\n'.join(hit_regions))

region_naming