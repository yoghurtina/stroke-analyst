import subprocess
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import cv2

fixed_image_path = '/home/ioanna/Documents/Thesis/src/registration/allenAtlas.nii.gz'
moving_image_path = '/home/ioanna/Documents/Thesis/src/registration/image.nii'
output_dir = '/home/ioanna/Documents/Thesis/src/registration'
dramms_path = "/opt/sbia/dramms-1.5.1/bin/dramms"
# NEEDS FIXING
def linear_registration(moving, fixed, ttype='similarity'):
    """
    Register grayscale images using SimpleITK.
    :param moving_path: string indicating path to moving image.
    :param fixed_path: string indicating path to fixed image.
    :param transformation_type: string indicating type of transformation to use.
        Must be one of: 'similarity', 'translation', 'rigid', 'affine'.
        Default is 'similarity'.
    :return: SimpleITK image of the registered moving image, transformation object,
        and the spatial referencing objects of the moving and fixed images.
    """

   # Read images
    moving = sitk.ReadImage(moving)
    fixed = sitk.ReadImage(fixed)

    # Get the image data as a numpy array
    moving = sitk.GetArrayFromImage(moving)
    fixed = sitk.GetArrayFromImage(fixed)

    # Add a new dimension with a size of 1 to the moving image
    if len(moving.shape) == 2:
        moving = moving.reshape((moving.shape[0], moving.shape[1], 1))

    # Convert numpy arrays to SimpleITK images
    moving_sitk = sitk.GetImageFromArray(moving)
    fixed_sitk = sitk.GetImageFromArray(fixed)

    # Check the size of the first dimension of the images and resize if necessary
    size0 = moving_sitk.GetSize()[0]
    if size0 < 4:
        factor = 4 // size0 + 1
        moving_sitk = sitk.Resample(moving_sitk, [size0 * factor, moving_sitk.GetSize()[1], moving_sitk.GetSize()[2]],
                                    sitk.Transform(), sitk.sitkLinear)

    size0 = fixed_sitk.GetSize()[0]
    if size0 < 4:
        factor = 4 // size0 + 1
        fixed_sitk = sitk.Resample(fixed_sitk, [size0 * factor, fixed_sitk.GetSize()[1], fixed_sitk.GetSize()[2]],
                                   sitk.Transform(), sitk.sitkLinear)
    
    # Intensity-based registration
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Set transform type
    if ttype == 'rigid':
        transformation_type = sitk.Euler3DTransform()
    elif ttype == 'similarity':
        transformation_type = sitk.Similarity3DTransform()
    elif ttype == 'translation':
        transformation_type = sitk.TranslationTransform(3)
    elif ttype == 'affine':
        transformation_type = sitk.AffineTransform(3)
    else:
        # default type : similarity transformation 
        transformation_type = sitk.Similarity3DTransform()
    
    registration_method.SetInitialTransform(transformation_type, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Perform registration
    final_transform = registration_method.Execute(sitk.Cast(fixed_sitk, sitk.sitkFloat32),
                                                   sitk.Cast(moving_sitk, sitk.sitkFloat32))
    
    # Apply transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    moving_reg_sitk = resampler.Execute(moving_sitk)

    # Remove the extra dimension from the moving image
    if moving_reg_sitk.GetDimension() == 3 and moving.shape[2] == 1:
        moving = np.squeeze(moving)


    # Store spatial referencing objects
    moving_ref = moving.GetSpatialReference()
    fixed_ref = fixed.GetSpatialReference()
    
    return sitk.GetArrayFromImage(moving_reg_sitk), final_transform, moving_ref, fixed_ref


linear_registration(moving_image_path, fixed_image_path)