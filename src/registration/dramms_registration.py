import os
from os.path import join as jp
from os.path import dirname as dn
from os.path import exists
from nilearn.image import load_img

import subprocess
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import SimpleITK as sitk

abs_dramms = os.path.abspath('/opt/sbia/dramms-1.5.1/bin/')

def affine(input_file, output_file, ref_file, search_direction=0):
    """This function finds an affine transformation using flirt. It calls the
    flirt program through the commmand line using the parameter -dof 6 that
    stands for rigid registration.

    Parameters:
    input_file (str): absolute path of the MRI file we want to register.
    output_file (str): absolute path of the registered MRI file name. In this
                       path the transformation file is also saved with file
                       extension .mat.
    ref_file (str): absolute path of the MRI file used as reference for the
                    registration (atlas space.)
    search_direction (int): the direction of search (default=0)
                            for left knees set to 90

    Returns:
    -----------
    """
    if not exists(dn(output_file)):
        os.makedirs(dn(output_file))

    search_direction = str(search_direction)
    dof = str(7)  # if this does not work try 12 or 7
    cost = 'normcorr'  # if this does not work try normmi or leastsq

    subcmd = ' -searchrx -' + search_direction + ' ' + search_direction + \
        ' -searchry -' + search_direction + ' ' + search_direction 
           
    command = 'flirt -in ' + input_file + ' -ref ' + ref_file + ' -out ' + \
        output_file + ' -omat ' + output_file[:-7] + '.mat -bins 256' + '-2D'
    os.system(command)


def create_landmarks_mask(image_file, landmark_coords):
    """Create a binary mask image with landmarks as non-zero values.

    Parameters:
    image_file (str): absolute path of the input MRI file.
    landmark_coords (list): list of landmark coordinates in the format of
                            [(x1, y1, z1), (x2, y2, z2), ...]

    Returns:
    mask (ndarray): binary mask image with the same dimensions as the input
                    MRI file, with landmarks as non-zero values.
    """
    # Load input MRI image
    
    image = sitk.ReadImage(image_file)

    # Create an empty binary mask with the same dimensions as the input image
    mask = np.zeros_like(image)

    # Set the voxel values at the landmark coordinates to non-zero
    for coord in landmark_coords:
        mask[coord] = 1

    return mask




def deform(input_file, output_file, ref_file, input_landmarks_file=None, ref_landmarks_file=None):
    """This function finds a deformable transformation using dramms. It calls
    the dramms program through the commmand line.

    Parameters:
    input_file (str): absolute path of the MRI file we want to register. It
                      must have been registered firstly using an affine or
                      rigid transformation.
    output_file (str): absolute path of the registered MRI file name. In this
                       path the transformation file is also saved with file
                       extension .nii.gz.
    ref_file (str): absolute path of the MRI file used as reference for the
                    registration (atlas space.)
    input_landmarks_file (str): absolute path of the file containing landmark
                               points in the input image space.
    ref_landmarks_file (str): absolute path of the file containing landmark
                             points in the reference image space.

    Returns:
    -----------
    """
    out_fol = dn(output_file)
    output_def = jp(out_fol, 'def_transf.nii.gz')

    command = abs_dramms + '/dramms -S ' + input_file + ' -T ' + ref_file + \
        ' -O ' + output_file + ' -D ' + output_def + ' -w 1 -a 0 -v -v' 
    if input_landmarks_file is not None and ref_landmarks_file is not None:
        command +='--landmarks' + ref_landmarks_file + ' ' + input_landmarks_file

    os.system(command)


def combine(input_aff, input_def, out_transf, original_mri, aff_mri):
    """This function combines 1 affine and 1 def transformation into a single
    file. The MRIs before the affine transformation and after the affine must
    also be provided.

    Parameters:
    input_aff (str): absolute path of the affine transformation file (.mat)
                     that we want to combine.
    input_def (str): absolute path of the deformable transformation file
                     (.nii.gz) that we want to combine.
    out_transf (str): absolute path of the combined transformation file
                     (.nii.gz).
    original_mri (str): absolute path of the MRI file before it was registered
                        with the affine transformation.
    aff_mri (str): absolute path of the MRI file after it was registered with
                   the affine transformation but not registered with the deform
                   transformation.

    Returns:
    -----------
    """

    command = abs_dramms + '/dramms-combine -c -f ' + original_mri + \
        ' -t ' + aff_mri + ' ' + input_aff + ' ' + input_def + \
        ' ' + out_transf

    os.system(command)


def inv_field(input_field, output_field):
    """This function calculates the inverse transformation of a deformable
    tranformation (deformation field with extension .nii.gz).

    Parameters:
    input_field (str): absolute path of the deformation field that we want to
                       invert.
    output_field (str): absolute path of the inverse deformation field.

    Returns:
    -----------
    """

    command = abs_dramms + '/dramms-defop -i ' + input_field + ' ' + \
        output_field

    os.system(command)


def transf_label(input_label, input_field, output_label, template_file):
    """This function applies a transformation to a labelmap.

    Parameters:
    input_label (str): absolute path of the labelmap file that we want to
                       transform.
    input_field (str): absolute path of the transformation that we want to
                       apply to the labelmap.
    output_label (str): absolute path of the transformed labelmap.

    Returns:
    -------------
    """

    if not exists(dn(output_label)):
        os.makedirs(dn(output_label))

    command = abs_dramms + '/dramms-warp ' + input_label + ' ' + \
        input_field + ' ' + output_label + ' -t ' + template_file + ' -n'

    os.system(command)


def resample(input_file, output_file, template_file, voxel_size):
    """This function changes the dimension and voxel size of a new image to
    match those of the target image, so that the inverse transformation can
    be calculated correctly.

    Parameters:
    input_file (str): absolute path of the target image.
    output_file (str): absolute path of the resulted resampled image.
    tempalte_file (str): absolute path of the template file used for
                         registration.
    voxel_size (array): the voxel size of the template image.

    Returns:
    -------------
    """

    if not exists(dn(output_file)):
        os.makedirs(dn(output_file))

    temp_img = load_img(template_file)
    dim = temp_img.shape
    command = abs_dramms + '/dramms-imgop -p ' + str(voxel_size[0]) + ',' + \
        str(voxel_size[1]) + ',' + \
        str(dim[0]) + ',' + str(dim[1]) + ',' + \
        input_file + ' ' + output_file

    os.system(command)

# fixed_image_path = '/home/ioanna/Documents/Thesis/src/registration/norm_fixed.nii'
# moving_image_path = '/home/ioanna/Documents/Thesis/src/registration/norm_moving.nii'
# output_dir = '/home/ioanna/Documents/Thesis/src/registration/'
dramms_path = "/opt/sbia/dramms-1.5.1/bin/"

# Read the input image
# moving_image = sitk.ReadImage('/home/ioanna/Documents/Thesis/src/registration/norm_moving.nii')
# fixed_image = sitk.ReadImage( '/home/ioanna/Documents/Thesis/src/registration/norm_fixed.nii')

# affine('/home/ioanna/Documents/Thesis/src/registration/norm_moving.nii', '/home/ioanna/Documents/Thesis/src/registration/results/dramms/affine/r_aff.nii.gz', '/home/ioanna/Documents/Thesis/src/registration/norm_fixed.nii')
# deform('/home/ioanna/Documents/Thesis/src/registration/norm_moving.nii', '/home/ioanna/Documents/Thesis/src/registration/results/dramms/def/r_def.nii.gz', '/home/ioanna/Documents/Thesis/src/registration/norm_fixed.nii')
# combine('/home/ioanna/Documents/Thesis/src/registration/dramms/affine/r_aff.mat', '/home/ioanna/Documents/Thesis/src/registration/dramms/def/r_def.nii.gz', 
#         '/home/ioanna/Documents/Thesis/src/registration/results/dramms/combine/comb.nii.gz', '/home/ioanna/Documents/Thesis/src/registration/norm_moving.nii',
#         '/home/ioanna/Documents/Thesis/src/registration/results/elastix/2.nii')

# resample('/home/ioanna/Documents/Thesis/src/registration/norm_moving.nii', '/home/ioanna/Documents/Thesis/src/registration/results/res.nii', '/home/ioanna/Documents/Thesis/src/registration/norm_fixed.nii', [0.5, 0.5])


deform('/home/ioanna/Documents/Thesis/src/registration/data/moving_flipped.nii', '/home/ioanna/Documents/Thesis/src/registration/reg_results/dramms/def/r_def.nii.gz', '/home/ioanna/Documents/Thesis/src/registration/data/fixed_flipped.nii')

import cv2
import numpy as np

# moving_image_array = sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path))
# fixed_image_array = sitk.GetArrayFromImage(sitk.ReadImage(fixed_image_path))

