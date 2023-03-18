import numpy as np
import nibabel as nib
from PIL import Image

img = Image.open("segmented_6.png")
img_as_array = np.asarray(img)

converted_array = np.array(img_as_array, dtype=np.float32) # You need to replace normal array by yours
affine = np.eye(4)
nifti_file = nib.Nifti1Image(converted_array, affine)

nib.save(nifti_file, "test.nii") # Here you put the path + the extionsion 'nii' or 'nii.gz'