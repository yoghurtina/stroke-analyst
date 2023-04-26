import SimpleITK as sitk
import itk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib


fixed_image_path = 'norm_fixed.nii'
# fixed_image_path = 'newimage.nii'
# fixed_image_path = 'fixed_bn2_68.nii'
# fixed_image_path = 'newfixed.nii'
fixed_image_path = 'fixed.jpg'

moving_image_path = 'norm_moving.nii'
# moving_image_path = 'moving.nii'
# moving_image_path = 'moving_bn2_68.nii'
# moving_image_path = 'newmoving.nii'
# moving_image_path = 'test.nii'
moving_image_path = 'moving.jpg'


def convert_to_grayscale(input_path):
    with Image.open(input_path) as img:
        img = img.convert('L')
    return img

# Convert image file to .nii format
def convert_image_nii(img_path, output):

    gray_img = convert_to_grayscale(img_path)
    img_as_array = np.asarray(gray_img)

    converted_array = np.array(img_as_array, dtype=np.float32) # You need to replace normal array by yours
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(converted_array, affine)
    output = output[:-4]

    return nifti_file
    nib.save(nifti_file, output+".nii") # Here you put the path + the extionsion 'nii' or 'nii.gz'

fixed_image = convert_image_nii(fixed_image_path, "fixed.jpg")

moving_image = convert_image_nii(moving_image_path, "moving.jpg")
print(fixed_image, moving_image)

def flip_nifti_image(nifti_img, output_path):
        # Load the NIfTI image
    # nifti_img = nib.load(nifti_path)
    img_data = nifti_img.get_fdata()

    # Flip the x and y axes of the image array
    # flipped_img_data = np.flip(img_data, axis=(0, 1))

    # Rotate the image array by 90 degrees clockwise
    rotated_img_data = np.rot90(img_data, k=1)

    # Create a new NIfTI image with the rotated data and the same affine transformation matrix
    rotated_nifti_img = nib.Nifti1Image(rotated_img_data, nifti_img.affine)

    # Save the rotated NIfTI image to the specified output path
    output_path = output_path + "_flipped.nii"  # Update the output path with "_flipped_rotated.nii" suffix
    nib.save(rotated_nifti_img, output_path)

nifti_path = "fixed.nii"
output_path = "test.nii"
flip_nifti_image(fixed_image, "fixed")
flip_nifti_image(moving_image, "moving")

moving_image = sitk.GetArrayFromImage(sitk.ReadImage("moving_flipped.nii"))
fixed_image = sitk.GetArrayFromImage(sitk.ReadImage("fixed_flipped.nii"))


plt.subplot(1, 2, 1)
plt.imshow(fixed_image, cmap='gray')
plt.title('Moving Image')

plt.subplot(1, 2, 2)
plt.imshow(moving_image, cmap='gray')
plt.title('Fixed Image')
plt.show()

# fixed_point_set_path = "fixedPointSet.pts"
# moving_point_set_path = "movingPointSet.pts"

# fixed_image = sitk.ReadImage(fixed_image_path)
# moving_image = sitk.ReadImage(moving_image_path)

# parameterMap["Metric"].append("CorrespondingPointsEuclideanDistanceMetric")

# elastixImageFilter = sitk.ElastixImageFilter()
# elastixImageFiltere = sitk.GetDefaultParameterMap("bspline")

# elastixImageFilter.AddParameter( 1, "Metric", "CorrespondingPointsEuclideanDistanceMetric" )


# # Compute the transformation
# elastixImageFilter = sitk.ElastixImageFilter()
# elastixImageFilter.SetFixedImage(fixed_image)
# elastixImageFilter.SetMovingImage(moving_image)
# elastixImageFilter.Execute()

# # Warp point set. The transformed points will be written to a file named
# # outputpoints.txt in the output directory determined by SetOutputDirectory()
# # (defaults to working directory). The moving image is needed for transformix
# # to correctly infer the dimensionality of the point set.
# transformixImageFilter = sitk.TransformixImageFilter()
# transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
# transformixImageFilter.SetFixedPointSetFileName("fixedPointSet.pts")
# result = transformixImageFilter.Execute()
# # result.SetRequestedRegion(result.GetLargestPossibleRegion())

# print(fixed_image.GetSize(), moving_image.GetSize(), result.GetSize(), fixed_image.GetOrigin(), moving_image.GetOrigin())

# sitk.WriteImage(result, 'tets.nii')

# print(type(result))

# result_a = sitk.GetArrayFromImage(result)
# # print(result_a)

# # import matplotlib.pyplot as plt

# # plt.imshow(result_a)
# # plt.show()


# RegImFilt = sitk.ElastixImageFilter()
# RegImFilt.LogToFileOn()
# RegImFilt.LogToConsoleOn()

# ParamMap = sitk.GetDefaultParameterMap('bspline')
# ParamMap['Transform'] = ['BSplineTransform']

# # Because of the increased complexity of the b-spline transform,
# # it is a good idea to run the registration a little longer to
# # ensure convergence
# ParamMap['MaximumNumberOfIterations'] = ['256']


# ParamMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
# # Need to have a metric that proceeds CorrespondingPointsEuclideanDistanceMetric - see [1]:
# ParamMap['Metric'] = ['AdvancedMattesMutualInformation', 'CorrespondingPointsEuclideanDistanceMetric']
# # I've also tried setting a small non-zero weight for AMMI but keeping the weight for CPEDM dominant:
# ParamMap['Metric0Weight'] = ['0.0']
# ParamMap['Metric1Weight'] = ['1.0']
# RegImFilt.SetParameterMap(ParamMap)

# RegImFilt.SetFixedImage(fixed_image)
# RegImFilt.SetMovingImage(moving_image)
# RegImFilt.SetFixedPointSetFileName(fixed_point_set_path)
# RegImFilt.SetMovingPointSetFileName(moving_point_set_path)
# RegImFilt.Execute()

# RegIm = RegImFilt.GetResultImage()

# sitk.WriteImage(RegIm, 'test.nii')

# plt.subplot(1, 3, 1)
# plt.imshow(sitk.GetArrayFromImage(fixed_image), cmap='gray')
# plt.title('Moving Image')

# plt.subplot(1, 3, 2)
# plt.imshow(sitk.GetArrayFromImage(moving_image), cmap='gray')
# plt.title('Fixed Image')

# plt.subplot(1,3,3)
# plt.imshow(sitk.GetArrayFromImage(RegIm), cmap = 'gray')
# plt.show()



