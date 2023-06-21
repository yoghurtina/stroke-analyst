import SimpleElastix as sitk

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

# Plot the moving and fixed images
plt.subplot(1, 2, 1)
plt.imshow(fixed_image_array, cmap='gray')
plt.title('Fixed Image')

plt.subplot(1, 2, 2)
plt.imshow(moving_image_array, cmap='gray')
plt.title('Fixed Image')

plt.show()


def demons_registration(fixed_image, moving_image, fixed_points = None, moving_points = None):
    print(fixed_image.GetSize(), moving_image.GetSize())
    print(fixed_image.GetDimension(), moving_image.GetDimension())
    print(fixed_image.GetOrigin(), moving_image.GetOrigin())
    print(fixed_image.GetSpacing(), moving_image.GetSpacing())
    print(fixed_image.GetNumberOfComponentsPerPixel(), moving_image.GetNumberOfComponentsPerPixel())

    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    print(transform_to_displacment_field_filter.GetSize())

    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform(2,sitk.sitkIdentity)))    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0) 
    
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(10) #intensities are equal if the difference is less than 10HU
        
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])    

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer    
    #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-7, convergenceWindowSize=20)
    registration_method.SetOptimizerScalesFromPhysicalShift()

     # Get the final transform from the registration result

    registered_image = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0)
        
    return registered_image   



result = demons_registration(fixed_image,moving_image)
print(result)
# registered_image = result.GetResultImage()


sitk.WriteImage(result, "/home/ioanna/Documents/Thesis/src/registration/reg_results/4.nii")


    
def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform = None, 
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors (list of lists or scalars): Shrink factors relative to the original image's size. When the list entry, 
                                                   shrink_factors[i], is a scalar the same factor is applied to all axes.
                                                   When the list entry is a list, shrink_factors[i][j] is applied to axis j.
                                                   This allows us to specify different shrink factors per axis. This is useful
                                                   in the context of microscopy images where it is not uncommon to have
                                                   unbalanced sampling such as a 512x512x8 image. In this case we would only want to 
                                                   sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
        smoothing_sigmas (list of lists or scalars): Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns: 
        SimpleITK.DisplacementFieldTransform
    """
        
    # Create initial displacement field at lowest resolution. 
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform, 
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_image.GetSize(),
                                                                       fixed_image.GetOrigin(),
                                                                       fixed_image.GetSpacing(),
                                                                       fixed_image.GetDirection())
        initial_displacement_field = registration_algorithm.Execute(fixed_image, 
                                                                moving_image, 
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.    
        initial_displacement_field = sitk.Resample (initial_displacement_field, fixed_image)
        initial_displacement_field = registration_algorithm.Execute(fixed_image, moving_image, initial_displacement_field)
        return sitk.DisplacementFieldTransform(initial_displacement_field)



# Define a simple callback which allows us to monitor the Demons filter's progress.
def iteration_callback(filter):
    print('\r{0}: {1:.2f}'.format(filter.GetElapsedIterations(), filter.GetMetric()), end='')


# Select a Demons filter and configure it.
demons_filter =  sitk.FastSymmetricForcesDemonsRegistrationFilter()
demons_filter.SetNumberOfIterations(20)
# Regularization (update field - viscous, total field - elastic).
demons_filter.SetSmoothDisplacementField(True)
demons_filter.SetStandardDeviations(2.0)

# Add our simple callback to the registration filter.
demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))



# Run the registration.
tx = multiscale_demons(registration_algorithm=demons_filter, 
                       fixed_image = fixed_image, 
                       moving_image = moving_image,
                       shrink_factors = [4,2],
                       smoothing_sigmas = [8,4])

print(tx)
# registered_image = result.GetResultImage()


# sitk.WriteImage(tx, "/home/ioanna/Documents/Thesis/src/registration/reg_results/4.nii")



def bspline_intra_modal_registration(
    fixed_image,
    moving_image,
    fixed_image_mask=None,
    fixed_points=None,
    moving_points=None,
):
    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    grid_physical_spacing = [4.0, 4.0, 4.0]  # A control point every 50mm
    image_physical_size = [
        size * spacing
        for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())
    ]
    mesh_size = [
        int(image_size / grid_spacing + 0.5)
        for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)
    ]

    initial_transform = sitk.BSplineTransformInitializer(
        image1=fixed_image, transformDomainMeshSize=mesh_size, order=3
    )
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5, numberOfIterations=100
    )

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    # if fixed_points and moving_points:
    #     registration_method.AddCommand(
    #         sitk.sitkStartEvent, rc.metric_and_reference_start_plot
    #     )
    #     registration_method.AddCommand(
    #         sitk.sitkEndEvent, rc.metric_and_reference_end_plot
    #     )
    #     registration_method.AddCommand(
    #         sitk.sitkIterationEvent,
    #         lambda: rc.metric_and_reference_plot_values(
    #             registration_method, fixed_points, moving_points
    #         ),
    #     )
    registration_method.Execute(fixed_image, moving_image)
    registered_image = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0)

    return registered_image


tx = bspline_intra_modal_registration(
    fixed_image=fixed_image,
    moving_image=moving_image)


print(tx)
print(type(tx))


sitk.WriteImage(tx, "/home/ioanna/Documents/Thesis/src/registration/reg_results/5.nii")

