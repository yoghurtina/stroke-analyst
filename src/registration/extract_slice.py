import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image


# NEEDS fixing

def extract_allen_slice(index):
    # load atlas
    size = [528, 320, 456]
    # VOL = 3-D matrix of atlas Nissl volume
    with open('atlasVolume.raw', 'rb') as file:
        VOL = np.fromfile(file, dtype=np.uint8, count=np.prod(size))
    VOL = np.reshape(VOL, size)

    atlas = [None] * np.size(VOL, 0)
    # empty = []
    # data = []
    for i in range(np.size(VOL, 0 )):
        atlas[i] = np.uint8(np.squeeze(VOL[i, :, :]))
    
    slice = atlas[index]
    plt.imshow(slice)
    plt.show()
    
    return slice


test = extract_allen_slice(200)
print(test)


img_as_array = np.asarray(test)

converted_array = np.array(img_as_array, dtype=np.float32) # You need to replace normal array by yours
affine = np.eye(4)
nifti_file = nib.Nifti1Image(converted_array, affine)

nib.save(nifti_file, "test.nii") # Here you put the path + the extionsion 'nii' or 'nii.gz'