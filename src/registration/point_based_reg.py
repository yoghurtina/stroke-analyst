import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib


fixed_image_path = '../data/norm_fixed.nii'
# fixed_image_path = 'newimage.nii'
# fixed_image_path = 'fixed_bn2_68.nii'
# fixed_image_path = 'newfixed.nii'
fixed_image_path = '../data/fixed.jpg'

moving_image_path = '../data/norm_moving.nii'
# moving_image_path = 'moving.nii'
# moving_image_path = 'moving_bn2_68.nii'
# moving_image_path = 'newmoving.nii'
# moving_image_path = 'test.nii'
moving_image_path = '../data/moving.jpg'


def convert_to_grayscale(input_path):
    with Image.open(input_path) as img:
        img = img.convert('L')
    return img

# Convert image file to .nii format
def convert_image_nii(img_path, output):

    gray_img = convert_to_grayscale(img_path)
    img_as_array = np.asarray(gray_img)

    converted_array = np.array(img_as_array, dtype=np.uint8) # You need to replace normal array by yours
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(converted_array, affine)
    output = output[:-4]

    return nifti_file
    nib.save(nifti_file, output+".nii") # Here you put the path + the extionsion 'nii' or 'nii.gz'

fixed_image = convert_image_nii(fixed_image_path, "../data/fixed.jpg")

moving_image = convert_image_nii(moving_image_path, "../data/moving.jpg")
print(fixed_image, moving_image)

def flip_nifti_image(nifti_img, output_path):
        # Load the NIfTI image
    # nifti_img = nib.load(nifti_path)
    img_data = nifti_img.get_fdata()

    # Flip the x and y axes of the image array
    # flipped_img_data = np.flip(img_data, axis=(0, 1))

    # Rotate the image array by 90 degrees clockwise
    rotated_img_data = np.fliplr(img_data)

    rotated_img_data = np.rot90(rotated_img_data, k=1)

    rotated_img_data = (rotated_img_data).astype(np.uint8)
    # Create a new NIfTI image with the rotated data and the same affine transformation matrix
    rotated_nifti_img = nib.Nifti1Image(rotated_img_data, nifti_img.affine)

    # Save the rotated NIfTI image to the specified output path
    output_path = output_path + "_flipped.nii"  
    nib.save(rotated_nifti_img, output_path)

    return rotated_nifti_img

nifti_path = "../data/fixed.nii"
output_path = "../data/test.nii"
flip_nifti_image(fixed_image, "../data/fixed")
re = flip_nifti_image(moving_image, "../data/moving")
print(re)

moving_image = sitk.GetArrayFromImage(sitk.ReadImage("../data/moving_flipped.nii"))
fixed_image = sitk.GetArrayFromImage(sitk.ReadImage("../data/fixed_flipped.nii"))


plt.subplot(1, 2, 1)
plt.imshow(fixed_image, cmap='gray')
plt.title('Moving Image')

plt.subplot(1, 2, 2)
plt.imshow(moving_image, cmap='gray')
plt.title('Fixed Image')
plt.show()

fixed_point_set_path = "../data/fixedPointSet.pts"
moving_point_set_path = "../data/movingPointSet.pts"

fixed_image = sitk.ReadImage("../data/fixed_flipped.nii")
moving_image = sitk.ReadImage("../data/moving_flipped.nii")


RegImFilt = sitk.ElastixImageFilter()
RegImFilt.LogToFileOn()
RegImFilt.LogToConsoleOn()

ParamMap = sitk.GetDefaultParameterMap('affine')
ParamMap['Transform'] = ['AffineTransform']

# Because of the increased complexity of the b-spline transform,
# it is a good idea to run the registration a little longer to
# ensure convergence
ParamMap['MaximumNumberOfIterations'] = ['1000']


ParamMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
# Need to have a metric that proceeds CorrespondingPointsEuclideanDistanceMetric - see [1]:
ParamMap['Metric'] = ['AdvancedMattesMutualInformation', 'CorrespondingPointsEuclideanDistanceMetric']
# I've also tried setting a small non-zero weight for AMMI but keeping the weight for CPEDM dominant:
ParamMap['Metric0Weight'] = ['0.0']
ParamMap['Metric1Weight'] = ['1.0']
RegImFilt.SetParameterMap(ParamMap)

RegImFilt.SetFixedImage(fixed_image)
RegImFilt.SetMovingImage(moving_image)
RegImFilt.SetFixedPointSetFileName(fixed_point_set_path)
RegImFilt.SetMovingPointSetFileName(moving_point_set_path)
RegImFilt.Execute()

RegIm = RegImFilt.GetResultImage()

plt.subplot(1, 3, 1)
plt.imshow(sitk.GetArrayFromImage(fixed_image), cmap='gray')
plt.title('Moving Image')

plt.subplot(1, 3, 2)
plt.imshow(sitk.GetArrayFromImage(moving_image), cmap='gray')
plt.title('Fixed Image')

plt.subplot(1,3,3)
plt.imshow(sitk.GetArrayFromImage(RegIm), cmap = 'gray')
plt.show()



