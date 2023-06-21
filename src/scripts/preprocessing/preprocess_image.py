from skimage import exposure
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np 
import cv2

# Read in the fixed image
# fixed_image_path = 'newimage.nii'
# fixed_image_path = 'newfixed.nii'
fixed_image_path = 'fixed3.nii'
fixed_image_1 = sitk.ReadImage(fixed_image_path)
fixed_image  = sitk.GetArrayFromImage(fixed_image_1)

# Read in the moving image
# moving_image_path = 'moving.nii'
# moving_image_path = 'newmoving.nii'
moving_image_path = 'moving3.nii'
moving_image_1 = sitk.ReadImage(moving_image_path)
moving_image = sitk.GetArrayFromImage(moving_image_1)



# Plot the moving and fixed images
plt.subplot(1, 2, 1)
plt.imshow(fixed_image, cmap='gray')
plt.title('Moving Image')

plt.subplot(1, 2, 2)
plt.imshow(moving_image, cmap='gray')
plt.title('Fixed Image')

plt.show()

def dpi_fixing(image, dpi = [300, 300]):
       
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


moving_dpi = dpi_fixing(moving_image_1)
fixed_dpi = dpi_fixing(fixed_image_1)

sitk.WriteImage(moving_dpi, 'dpi_moving.nii')
sitk.WriteImage(fixed_dpi, 'dpi_fixed.nii')

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


# resize moving image to match the dimensions of the fixed one
resized_moving_image = resizeAndPad(moving_dpi, fixed_dpi.GetSize(), direction=fixed_dpi.GetDirection())
# save the resized moving image
# sitk.WriteImage(moving_image, 'test.nii')



# Plot the moving and fixed images
plt.subplot(1, 2, 1)
plt.imshow(sitk.GetArrayFromImage(fixed_dpi), cmap='gray')
plt.title('Fixed Image')

plt.subplot(1, 2, 2)
plt.imshow(sitk.GetArrayFromImage(resized_moving_image), cmap='gray')
plt.title('Fixed Image')

plt.show()



path = "bp1_52.jpg"
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
    sitk.WriteImage(fixed_norm_image, 'norm_fixed.nii')

    moving_norm_image =sitk.GetImageFromArray(moving_norm)
    moving_norm_image.SetOrigin((0.0, 0.0, 0.0)) # Replace with appropriate origin values
    moving_norm_image.SetSpacing((1.0, 1.0, 1.0)) # Replace with appropriate spacing values
    moving_norm_image.SetDirection(np.identity(2).ravel()) # Replace with appropriate direction matrix
    moving_norm_image=sitk.Cast(moving_norm_image, sitk.sitkFloat32)
    sitk.WriteImage(moving_norm_image, 'norm_moving.nii')

    # print(fixed_norm.shape, moving_norm.shape)
    return fixed_norm, moving_norm, fixed_norm_image, moving_norm_image


fixed_norm, moving_norm, fixed_norm_image, moving_norm_image = normalization(sitk.GetArrayFromImage(fixed_dpi), sitk.GetArrayFromImage(resized_moving_image))


# Plot the moving and fixed images
plt.subplot(1, 2, 1)
plt.imshow(fixed_norm, cmap='gray')
plt.title('Moving Image')

plt.subplot(1, 2, 2)
plt.imshow(moving_norm, cmap='gray')
plt.title('Fixed Image')

plt.show()


