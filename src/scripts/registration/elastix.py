import SimpleITK as sitk
import matplotlib.pyplot as plt 

fixed_image_path = 'data/fixed.nii'
# fixed_image_path = 'newimage.nii'
# fixed_image_path = 'fixed_bn2_68.nii'
# fixed_image_path = 'newfixed.nii'
# fixed_image_path = 'fixedt.jpg'



moving_image_path = 'data/moving.nii'
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

def resampler(moving_image, fixed_image):
    # Get the fixed image spacing
    fixed_spacing = fixed_image.GetSpacing()

    # Resample the moving image to have the same spacing as the fixed image
    resampled_image = sitk.Resample(moving_image, fixed_image.GetSize(), sitk.Transform(), sitk.sitkLinear, moving_image.GetOrigin(), fixed_spacing, moving_image.GetDirection(), 0.0, moving_image.GetPixelID())

    # You can save the resampled image to a new file if needed
    sitk.WriteImage(resampled_image, "/home/ioanna/Documents/Thesis/src/registration/reg_results/elastix/resampled_image.nii")
    return resampled_image

# moving_image = resampler(moving_image, fixed_image)


def rigid(fixed_image_path, moving_image_path):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOn()

    elastixImageFilter.SetFixedImage(sitk.ReadImage(fixed_image_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(moving_image_path))
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/reg_results/1.nii")

    return elastixImageFilter.GetResultImage()

def affine(fixed_image_path, moving_image_path):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOn()

    elastixImageFilter.SetFixedImage(sitk.ReadImage(fixed_image_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(moving_image_path))
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(),  "/home/ioanna/Documents/Thesis/src/registration/reg_results/2.nii")

    return elastixImageFilter.GetResultImage()

def non_rigid(fixed_image_path, moving_image_path):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOn()

    elastixImageFilter.SetFixedImage(sitk.ReadImage(fixed_image_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(moving_image_path))

    # Configure the parameter map for affine registration
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))

    # Configure the parameter map for B-spline registration
    bspline_parameter_map = sitk.GetDefaultParameterMap("bspline")

    # Modify B-spline parameters for smoother deformation and image pyramid
    bspline_parameter_map['ImagePyramidSchedule'] = ['32, 16, 8, 6, 4, 2, 1']  # Image pyramid schedule with decreasing scale factors
    bspline_parameter_map['BSplineSmoothingSigma'] = ['10']  # Increase the smoothing sigma value for smoother deformation
    bspline_parameter_map['FinalGridSpacingInPhysicalUnits'] = ['32, 16, 10, 8, 6, 4, 2, 1']
        # bspline_parameter_map['FinalGridSpacingInPhysicalUnits'] = ['32, 16, 10, 8, 6, 4, 2']

    parameterMapVector.append(bspline_parameter_map)

    # Set the parameter map and execute the registration
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    # Save the resulting image
    sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/reg_results/3.nii")

    return elastixImageFilter.GetResultImage()



# rigid(fixed_image_path, moving_image_path)
# affine(fixed_image_path, moving_image_path)


re = non_rigid(fixed_image_path, moving_image_path)
# Plot the moving and fixed images
plt.subplot(1, 3, 1)
plt.imshow(fixed_image_array, cmap='gray')
plt.title('Fixed Image')

plt.subplot(1, 3, 2)
plt.imshow(moving_image_array, cmap='gray')
plt.title('Fixed Image')

plt.subplot(1, 3, 3)
plt.imshow(sitk.GetArrayFromImage(re), cmap='gray')
plt.title('Fixed Image')

plt.show()


# parameterMap = sitk.GetDefaultParameterMap('bspline')

# # Use a non-rigid transform instead of a translation transform
# parameterMap['Transform'] = ['BSplineTransform']

# # Because of the increased complexity of the b-spline transform,
# # it is a good idea to run the registration a little longer to
# # ensure convergence
# parameterMap['MaximumNumberOfIterations'] = ['10000']

# parameterMap[ "Registration" ] =  ["MultiMetricMultiResolutionRegistration" ]
# parameterMap["FixedImagePyramidSchedule"] = ["1", "0.5", "0.2", "0.1", "0.05", "0.01"]  # Adjust pyramid levels for desired level of smoothness
# parameterMap["MovingImagePyramidSchedule"] = ["1", "0.5", "0.2", "0.1", "0.05", "0.01"]
# # Set the parameter map in the Elastix image filter
# elastixImageFilter.SetParameterMap(parameterMap)

# elastixImageFilter.Execute()
# sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/reg_results/4.nii")


