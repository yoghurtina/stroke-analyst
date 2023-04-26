import SimpleITK as sitk
import matplotlib.pyplot as plt 

fixed_image_path = 'norm_fixed.nii'
# fixed_image_path = 'newimage.nii'
# fixed_image_path = 'fixed_bn2_68.nii'
# fixed_image_path = 'newfixed.nii'
# fixed_image_path = 'fixedt.jpg'



moving_image_path = 'norm_moving.nii'
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
    sitk.WriteImage(resampled_image, "/home/ioanna/Documents/Thesis/src/registration/results/elastix/resampled_image.nii")
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
sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/results/elastix/1.nii")

parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/results/elastix/2.nii")

parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/results/elastix/3.nii")

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
sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/ioanna/Documents/Thesis/src/registration/results/elastix/4.nii")


# import os

# def deform(input_file, output_file, ref_file):
#     """This function finds a deformable transformation using elastix. It calls
#     the elastix program through SimpleITK.

#     Parameters:
#     input_file (str): absolute path of the MRI file we want to register. It
#                       must have been registered firstly using an affine or
#                       rigid transformation.
#     output_file (str): absolute path of the registered MRI file name. In this
#                        path the transformation file is also saved with file
#                        extension .nii.gz.
#     ref_file (str): absolute path of the MRI file used as reference for the
#                     registration (atlas space.)
#     input_landmarks_file (str): absolute path of the file containing landmark
#                                points in the input image space.
#     ref_landmarks_file (str): absolute path of the file containing landmark
#                              points in the reference image space.

#     Returns:
#     -----------
#     """
#     # Load input and reference images
#     input_image = sitk.ReadImage(input_file)
#     ref_image = sitk.ReadImage(ref_file)

#     # Create registration object
#     registration = sitk.ElastixImageFilter()

#     # Set input and reference images
#     registration.SetFixedImage(ref_image)
#     registration.SetMovingImage(input_image)

#     # Set parameter map for deformable registration
#     parameter_map = sitk.GetDefaultParameterMap("bspline")
#     parameter_map['FinalBSplineInterpolationOrder'] = ['3']
#     registration.SetParameterMap(parameter_map)

#     # Set output directory for transformation and results
#     out_fol = os.path.dirname(output_file)
#     registration.SetOutputDirectory(out_fol)

#     # Execute registration
#     registration.Execute()

#     # Get registered image and transformation
#     registered_image = registration.GetResultImage()
#     transformation = registration.GetTransformParameterMap()

#     # Save registered image and transformation
#     sitk.WriteImage(registered_image, output_file)
#     sitk.WriteParameterFile(transformation, os.path.join(out_fol, 'deformation.txt'))

# deform('/home/ioanna/Documents/Thesis/src/registration/norm_moving.nii', '/home/ioanna/Documents/Thesis/src/registration/results/elastix/def.nii.gz', '/home/ioanna/Documents/Thesis/src/registration/norm_fixed.nii')
