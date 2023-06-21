import numpy as np
import nibabel as nib
from PIL import Image
import SimpleITK as sitk

# This script converts jpg, png etc in .nii
# Also converts .raw to .nii.gz (atlas)

def convert_to_grayscale(input_path):
    with Image.open(input_path) as img:
        img = img.convert('L')
    return img

# Convert image file to .nii format
def convert_image_nii(img_path):

    gray_img = convert_to_grayscale(img_path)
    img_as_array = np.asarray(gray_img)

    converted_array = np.array(img_as_array, dtype=np.uint8) # You need to replace normal array by yours
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(converted_array, affine)

    nib.save(nifti_file, "fixed3.nii") # Here you put the path + the extionsion 'nii' or 'nii.gz'



# Convert .raw to .nii.gz (accepted from DRAMMS)
def convert_allen_nii(atlas_path):
    # Define the size of the image
    size1 = (528, 320, 456)

    # Load the raw binary data into a numpy array
    with open(atlas_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        
    # Reshape the numpy array to the desired size
    data = data.reshape(size1[::-1]) # swap the order of dimensions

    # Convert the numpy array to a SimpleITK image object
    image = sitk.GetImageFromArray(data)

    # Save the image to disk
    sitk.WriteImage(image, "allenAtlas.nii.gz")

# convert_image_nii('sagittal0153.png')
convert_image_nii("data/fixed3.jpg")
# convert_image_nii("data/moving1.jpg")
# 
# convert_allen_nii('atlasVolume.raw')