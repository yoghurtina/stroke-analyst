import numpy as np
import nibabel as nib
from PIL import Image
import SimpleITK as sitk
from skimage import exposure
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np 
import cv2

def convert_to_grayscale(input_path):
    with Image.open(input_path) as img:
        img = img.convert('L')
    return img

# Convert image file to .nii format
def convert_image_nii(img_path, save_path):

    gray_img = convert_to_grayscale(img_path)
    img_as_array = np.asarray(gray_img)

    converted_array = np.array(img_as_array, dtype=np.uint8) # You need to replace normal array by yours
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(converted_array, affine)

    nib.save(nifti_file, save_path) # Here you put the path + the extionsion 'nii' or 'nii.gz'


def dpi_fixing(image_path, dpi = [600, 600]):
    image = sitk.ReadImage(image_path)
    # Get the current spacing and size of the image
    spacing = image.GetSpacing()
    size = image.GetSize()

    # Calculate the new spacing based on the desired DPI
    new_spacing = (spacing[0]*size[0]/dpi[0], spacing[1]*size[1]/dpi[1], 1)

    # Resample the image with the new spacing
    resampled_img = sitk.Resample(image, [int(size[0]*spacing[0]/new_spacing[0]), 
                                          int(size[1]*spacing[1]/new_spacing[1]), 1],
                                         sitk.Transform(), sitk.sitkLinear, image.GetOrigin(),
                                           new_spacing, image.GetDirection(), 0, image.GetPixelID())
    resampled_img.SetOrigin((0.0, 0.0))
    resampled_img.SetSpacing((1.0, 1.0))
    # Save the image with the new DPI
    return resampled_img

def resizeAndPad(img, size, padColor=0, direction=None):
    img = sitk.GetArrayFromImage(img)
    h, w = img.shape[:2]
    sh, sw = size[::-1]

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    scaled_img = sitk.GetImageFromArray(scaled_img)

    # set direction
    if direction is not None:
        print(direction)
        scaled_img.SetDirection(direction)

    return scaled_img


# Normalize image intensities to ensure they have similar brightness and contrast.
def normalization(fixed_image, moving_image):
    # array_moving = sitk.GetArrayFromImage(moving_image)
    # array_fixed =sitk.GetArrayFromImage(fixed_image)

    fixed_norm = exposure.equalize_hist(fixed_image)
    moving_norm = exposure.equalize_hist(moving_image)
    fixed_norm_image = sitk.GetImageFromArray(fixed_norm)

    fixed_norm_image.SetOrigin((0.0, 0.0, 0.0)) # Replace with appropriate origin values
    fixed_norm_image.SetSpacing((1.0, 1.0, 1.0)) # Replace with appropriate spacing values
    fixed_norm_image.SetDirection(np.identity(2).ravel()) # Replace with appropriate direction matrix
    fixed_norm_image=sitk.Cast(fixed_norm_image,sitk.sitkFloat32 )
    # sitk.WriteImage(fixed_norm_image, 'norm_fixed.nii')

    moving_norm_image =sitk.GetImageFromArray(moving_norm)
    moving_norm_image.SetOrigin((0.0, 0.0, 0.0)) # Replace with appropriate origin values
    moving_norm_image.SetSpacing((1.0, 1.0, 1.0)) # Replace with appropriate spacing values
    moving_norm_image.SetDirection(np.identity(2).ravel()) # Replace with appropriate direction matrix
    moving_norm_image=sitk.Cast(moving_norm_image, sitk.sitkFloat32)
    # sitk.WriteImage(moving_norm_image, 'norm_moving.nii')

    # print(fixed_norm.shape, moving_norm.shape)
    return fixed_norm, moving_norm, fixed_norm_image, moving_norm_image



def convert_to_jpg(image, save_path):
    # Convert the SimpleITK image to a NumPy array
    image_array = sitk.GetArrayFromImage(image)

    # Convert the NumPy array to a 8-bit unsigned integer
    numpy_array = image_array.astype(np.uint8)

    # Create a PIL image from the NumPy array
    pil_image = Image.fromarray(numpy_array)

    # Save the PIL image as a JPEG file
    pil_image.save(save_path)

