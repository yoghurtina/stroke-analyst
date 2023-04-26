import SimpleElastix as sitk

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


sitk.WriteImage(result, "/home/ioanna/Documents/Thesis/src/registration/results/elastix/4.nii")


    
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

