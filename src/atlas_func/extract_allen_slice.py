import numpy as np
import nibabel as nib
import SimpleITK as sitk
import math

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

print(find_coordinates(-2.28e-3))

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

temp = extract_allen_slice(308)
print(temp)
image = sitk.GetImageFromArray(temp)

sitk.WriteImage(image, "test.nii")

