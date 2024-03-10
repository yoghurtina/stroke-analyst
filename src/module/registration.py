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

# # moving_image = resampler(moving_image, fixed_image)

def rigid(fixed_image_path, moving_image_path, output_path="src/temp/registration/rigid_result.nii"):
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_image, moving_image)
    result_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    sitk.WriteImage(result_image, output_path)
    return result_image

def affine(fixed_image_path, moving_image_path, output_path="src/temp/registration/affine_result.nii"):
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.AffineTransform(2), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_image, moving_image)
    result_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    sitk.WriteImage(result_image, output_path)
    return result_image

def non_rigid(fixed_image_path, moving_image_path, output_path="src/temp/registration/non_rigid_result.nii"):
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Initialize B-Spline transform
    transform = sitk.BSplineTransformInitializer(fixed_image, [8, 8, 8], order=3)

    # Set up the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Use Mattes Mutual Information as the metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Use linear interpolation
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Setup for the optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution framework setup
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the initial transform
    registration_method.SetInitialTransform(transform, inPlace=False)

    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the final transform to the moving image
    result_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Save the result
    sitk.WriteImage(result_image, output_path)

    return result_image


def advanced_non_rigid(fixed_image_path, moving_image_path, output_path="src/temp/registration/non_rigid_advanced.nii"):
    # Read the fixed and moving images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Initial transformation (affine) for alignment
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_image, 
                                                          sitk.AffineTransform(fixed_image.GetDimension()),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Multi-resolution framework for image registration
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01, seed=1)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                      numberOfIterations=100, 
                                                      convergenceMinimumValue=1e-6, 
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for multi-resolution strategy
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Use the initial affine transform and then refine with B-Splines
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Final B-Spline transformation (non-rigid)
    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacement_field_filter.SetReferenceImage(fixed_image)
    transform_to_displacement_field_filter.SetInterpolator(sitk.sitkLinear)

    bspline_transform = sitk.BSplineTransformInitializer(fixed_image, 
                                                         transformDomainMeshSize=[8,8,8], 
                                                         order=3)
    registration_method.SetInitialTransform(bspline_transform, inPlace=False)

    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the final transform to resample the moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    resampler.SetDefaultPixelValue(100)
    resampled_image = resampler.Execute(moving_image)

    # Save the result
    sitk.WriteImage(resampled_image, output_path)

    return output_path
