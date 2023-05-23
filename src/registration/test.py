import SimpleITK as sitk
import os
import numpy as np


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


def register_img(fixed_arr, 
                 moving_arr,
                use_affine = True,
                use_mse = True,
                brute_force = True,
                show_transform = True):
    fixed_image = sitk.GetImageFromArray(fixed_arr)
    moving_image = sitk.GetImageFromArray(moving_arr)
    transform = sitk.AffineTransform(2) if use_affine else sitk.ScaleTransform(2)
    initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                      moving_image, 
                                                      transform, 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    ff_img = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mv_img = sitk.Cast(moving_image, sitk.sitkFloat32)
    registration_method = sitk.ImageRegistrationMethod()
    if use_mse:
        registration_method.SetMetricAsMeanSquares()
    else:
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    if brute_force:
        sample_per_axis = 12
        registration_method.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
        # Utilize the scale to set the step size for each dimension
        registration_method.SetOptimizerScales([2.0*3.14/sample_per_axis, 1.0,1.0])
    else:
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.25)

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                      numberOfIterations=200, 
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)
    # Scale the step size differently for each parameter, this is critical!!!
    registration_method.SetOptimizerScalesFromPhysicalShift() 

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform_v1 = registration_method.Execute(ff_img, 
                                                     mv_img)
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.     
    resample.SetInterpolator(sitk.sitkBSpline)  
    resample.SetTransform(final_transform_v1)
    if show_transform:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
        xx, yy = np.meshgrid(range(moving_arr.shape[0]), range(moving_arr.shape[1]))
        test_pattern = (((xx % 40)>30)|((yy % 40)>30)).astype(np.float32)
        ax1.imshow(test_pattern, cmap = 'bone_r')
        ax1.set_title('Test Pattern')
        test_pattern_img = sitk.GetImageFromArray(test_pattern)
        skew_pattern = sitk.GetArrayFromImage(resample.Execute(test_pattern_img))
        ax2.imshow(skew_pattern, cmap = 'bone_r')
        ax2.set_title('Registered Pattern')
        plt.show()
    return sitk.GetArrayFromImage(resample.Execute(moving_image))



re = register_img(fixed_image_array, moving_image_array, use_affine=False,brute_force=False ,show_transform=True)
print(re)

def register_img_generic(fixed_arr, 
                 moving_arr,
                registration_func,
                        show_transform = True):
    fixed_image = sitk.GetImageFromArray(fixed_arr)
    moving_image = sitk.GetImageFromArray(moving_arr)
    ff_img = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mv_img = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    final_transform_v1 = registration_func(ff_img, mv_img)
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.     
    resample.SetInterpolator(sitk.sitkBSpline)  
    resample.SetTransform(final_transform_v1)

    if show_transform:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
        xx, yy = np.meshgrid(range(moving_arr.shape[0]), range(moving_arr.shape[1]))
        test_pattern = (((xx % 40)>30)|((yy % 40)>30)).astype(np.float32)
        ax1.imshow(test_pattern, cmap = 'bone_r')
        ax1.set_title('Test Pattern')
        test_pattern_img = sitk.GetImageFromArray(test_pattern)
        skew_pattern = sitk.GetArrayFromImage(resample.Execute(test_pattern_img))
        ax2.imshow(skew_pattern, cmap = 'bone_r')
        ax2.set_title('Registered Pattern')
    return sitk.GetArrayFromImage(resample.Execute(moving_image))


def bspline_intra_modal_registration(fixed_image, moving_image, grid_physical_spacing =  [15.0]*3):
    registration_method = sitk.ImageRegistrationMethod()
    # Determine the number of BSpline control points using the physical spacing we want for the control grid. 
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    print('Using Mesh Size', mesh_size)
    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    registration_method.SetInitialTransform(initial_transform)
        
    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [1,1,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[20,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)

    # registration_method.SetOptimizerScalesFromPhysicalShift() 

    # registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform_v1 = registration_method.Execute(fixed_image, 
                                                     moving_image)

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    
    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.     
    resample.SetInterpolator(sitk.sitkBSpline)  
    resample.SetTransform(final_transform_v1)
    
    return sitk.GetArrayFromImage(resample.Execute(moving_image))

    # return registration_method.Execute(fixed_image, moving_image)

# re = bspline_intra_modal_registration(fixed_image, moving_image)

re = register_img_generic(fixed_image_array, re, registration_func=bspline_intra_modal_registration(fixed_image, moving_image))


print(re)
print(type(re))
sitk.WriteImage(sitk.GetImageFromArray(re), "/home/ioanna/Documents/Thesis/src/registration/reg_results/test.nii")
