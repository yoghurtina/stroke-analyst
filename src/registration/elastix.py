import SimpleITK as sitk
import matplotlib.pyplot as plt 

fixed_image_path = 'data/fixed_flipped.nii'
# fixed_image_path = 'newimage.nii'
# fixed_image_path = 'fixed_bn2_68.nii'
# fixed_image_path = 'newfixed.nii'
# fixed_image_path = 'fixedt.jpg'



moving_image_path = 'data/moving_flipped.nii'
# moving_image_path = 'moving.nii'
# moving_image_path = 'moving_bn2_68.nii'
# moving_image_path = 'newmoving.nii'
# moving_image_path = 'test.nii'
# moving_image_path = 'test.jpg'



fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

print(fixed_image.GetSize(), moving_image.GetSize())

fixed_image_array   = sitk.GetArrayFromImage(fixed_image)
moving_image_array  = sitk.GetArrayFromImage(moving_image)

# Plot the moving and fixed images
plt.subplot(1, 2, 1)
plt.imshow(fixed_image_array, cmap='gray')
plt.title('Fixed Image')

plt.subplot(1, 2, 2)
plt.imshow(moving_image_array, cmap='gray')
plt.title('Fixed Image')

plt.show()



def resampler(moving_image, fixed_image):
    # Get the fixed image spacing
    fixed_spacing = fixed_image.GetSpacing()

    # Resample the moving image to have the same spacing as the fixed image
    resampled_image = sitk.Resample(moving_image, fixed_image.GetSize(), sitk.Transform(), sitk.sitkLinear, moving_image.GetOrigin(), fixed_spacing, moving_image.GetDirection(), 0.0, moving_image.GetPixelID())

    # You can save the resampled image to a new file if needed
    sitk.WriteImage(resampled_image, "/home/ioanna/Documents/Thesis/src/registration/reg_results/elastix/resampled_image.nii")
    return resampled_image

# moving_image = resampler(moving_image, fixed_image)

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.LogToFileOn()

elastixImageFilter.SetFixedImage(fixed_image)
elastixImageFilter.SetMovingImage(moving_image)

elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
parameterMapVector = sitk.VectorOfParameterMap()

parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/reg_results/elastix/1.nii")

parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/reg_results/elastix/2.nii")

parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/reg_results/elastix/3.nii")

parameterMap = sitk.GetDefaultParameterMap('affine')

# Use a non-rigid transform instead of a translation transform
parameterMap['Transform'] = ['AffineTransform']

# Because of the increased complexity of the b-spline transform,
# it is a good idea to run the registration a little longer to
# ensure convergence
parameterMap['MaximumNumberOfIterations'] = ['10000']

parameterMap[ "Registration" ] =  ["MultiResolutionRegistration" ]
parameterMap["FixedImagePyramidSchedule"] = ["1", "0.5", "0.2", "0.1", "0.05", "0.01"]  # Adjust pyramid levels for desired level of smoothness
parameterMap["MovingImagePyramidSchedule"] = ["1", "0.5", "0.2", "0.1", "0.05", "0.01"]
# Set the parameter map in the Elastix image filter
elastixImageFilter.SetParameterMap(parameterMap)

elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/reg_results/elastix/4.nii")

