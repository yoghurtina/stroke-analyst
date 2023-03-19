import subprocess
import numpy as np
import SimpleITK as sitk
import nibabel as nib

def dramms_2dregistration(target_path, moving_path, out_path, file_name, dramms_path, *args):
    # Convert paths to strings
    target_path = str(target_path)
    moving_path = str(moving_path)
    out_path_dfield = str(out_path) + '/dfield.nii.gz'
    if len(args) > 0:
        out_path_dfield = str(args[0]) + '_dfield.nii.gz'
    out_path = str(out_path) + '/registered_' + str(file_name) + '_to_average.nii'

    # Keep registration info
    reg_info = {}
    reg_info['target_img_path'] = target_path
    reg_info['moving_img_path'] = moving_path
    reg_info['regi_output_path_dfield'] = out_path_dfield
    reg_info['regi_output_path'] = out_path

    # # Load the images using SimpleITK
    # target_img = sitk.ReadImage(target_path)
    # moving_img = sitk.ReadImage(moving_path)

    sett = "-w1 -a0 -c1"  # < -c 1 > for lesions and cuts..

    command = [
        str(dramms_path) + "/dramms",
        "-S", moving_path,
        "-T", target_path,
        "-O", out_path,
        "-D", out_path_dfield,
        sett
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    print(result)

# # Save the registered image and deformation field using SimpleITK
#     dfield = sitk.ReadImage(out_path_dfield)
#     reg_img = sitk.ReadImage(out_path)
#     sitk.WriteImage(dfield, out_path_dfield)
#     sitk.WriteImage(reg_img, out_path)
    
    return reg_info


fixed_image_path = '/home/ioanna/Documents/Thesis/src/registration/allenAtlas.nii.gz'
moving_image_path = '/home/ioanna/Documents/Thesis/src/registration/image.nii'
output_dir = '/home/ioanna/Documents/Thesis/src/registration'
dramms_path = "/opt/sbia/dramms-1.5.1/bin/"

info=dramms_2dregistration(moving_image_path, moving_image_path, output_dir, "test", dramms_path)   
print(info)

