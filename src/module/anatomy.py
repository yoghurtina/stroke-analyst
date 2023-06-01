import SimpleITK as sitk
import matplotlib.pyplot as plt 


def resampler(moving_image, fixed_image):
    # Get the fixed image spacing
    fixed_spacing = fixed_image.GetSpacing()
    # Resample the moving image to have the same spacing as the fixed image
    resampled_image = sitk.Resample(moving_image, fixed_image.GetSize(), sitk.Transform(), sitk.sitkLinear, moving_image.GetOrigin(), fixed_spacing, moving_image.GetDirection(), 0.0, moving_image.GetPixelID())
    # You can save the resampled image to a new file if needed
    # sitk.WriteImage(resampled_image, "/home/ioanna/Documents/Thesis/src/registration/reg_results/elastix/resampled_image.nii")
    return resampled_image

# moving_image = resampler(moving_image, fixed_image)


def rigid(fixed_image_path, moving_image_path):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOn()

    elastixImageFilter.SetFixedImage(sitk.ReadImage(fixed_image_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(moving_image_path))
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), "src/temp/anatomy/rigid.nii")

    return elastixImageFilter.GetResultImage()

def affine(fixed_image_path, moving_image_path):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOn()

    elastixImageFilter.SetFixedImage(sitk.ReadImage(fixed_image_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(moving_image_path))
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(),  "src/temp/anatomy/affine.nii")

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
    sitk.WriteImage(elastixImageFilter.GetResultImage(), "src/temp/anatomy/non_rigid.nii")

    return elastixImageFilter.GetResultImage()

