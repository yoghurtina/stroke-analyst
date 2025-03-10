# import streamlit as st
# from streamlit_cropper import st_cropper
# import os
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import SimpleITK as sitk

# from module.reg_preprocessing import convert_image_nii, dpi_fixing, convert_to_grayscale, convert_to_jpg
# from module.registration import rigid, non_rigid, affine
# from module.normalization import equalize_this
# import cv2

# col1, col2 = st.columns(2)

# def save_uploadedfile(uploadedfile, path):
#     with open(os.path.join(path),"wb") as f:
#         f.write(uploadedfile.getbuffer())
#     return True



# def reg_results():

#     uploaded_in_previous_step = Image.open("src/temp/alignment/aligned_section.jpg").convert('L')
#     # equalized = equalize_this("src/temp/alignment/aligned_section.jpg")
#     # cv2.imwrite("src/temp/registration/equalized_section.jpg", equalized)
#     # uploaded_in_previous_step = Image.open("src/temp/registration/equalized_section.jpg")
#     uploaded_array = np.array(uploaded_in_previous_step)
#     img_height, img_width = uploaded_array.shape

    
#     selected_atlas_section = Image.open("src/temp/mapping/mapped_allen_section.jpg")
#     st.image([uploaded_in_previous_step, selected_atlas_section], caption=["Moving Image", "Fixed Image"], width=300)

#     moving_image_path = "src/temp/registration/equalized_section.jpg"
#     fixed_image_path =  "src/temp/mapping/mapped_allen_section.jpg"

#     convert_image_nii(moving_image_path, "src/temp/registration/moving.nii")
#     convert_image_nii(fixed_image_path,  "src/temp/registration/fixed.nii")

#     moving_nii_path = "src/temp/registration/moving.nii"
#     fixed_nii_path =  "src/temp/registration/fixed.nii"

#     resampled_moving = dpi_fixing(moving_nii_path,fixed_nii_path)
#     resampled_fixed = dpi_fixing(fixed_nii_path, fixed_nii_path)

#     sitk.WriteImage(resampled_moving, "src/temp/registration/resampled_moving.nii" )
#     sitk.WriteImage(resampled_fixed,  "src/temp/registration/resampled_fixed.nii")

#     resampled_moving_path = "src/temp/registration/resampled_moving.nii"
#     resampled_fixed_path =  "src/temp/registration/resampled_fixed.nii"

#     rigid_result =          rigid(resampled_fixed_path, resampled_moving_path)
#     affine_result =        affine(resampled_fixed_path, resampled_moving_path)
#     non_rigid_result =  non_rigid(resampled_fixed_path, resampled_moving_path)

#     convert_to_jpg(rigid_result, "src/temp/registration/rigid.jpg")
#     convert_to_jpg(affine_result, "src/temp/registration/affine.jpg" )
#     convert_to_jpg(non_rigid_result, "src/temp/registration/non_rigid.jpg")

#     st.image([Image.open("src/temp/registration/rigid.jpg"), 
#             Image.open("src/temp/registration/affine.jpg"), 
#             Image.open("src/temp/registration/non_rigid.jpg")], width=200, caption=["Rigid Registration", "Affine Registration", "Non-rigid Registration"])


# st.header("Registration Results")

# def hem_reg_results():
#     uploaded_in_previous_step = Image.open("src/temp/cropper/stroked_hemisphere.jpg")
#     equalized = equalize_this("src/temp/alignment/aligned_section.jpg")
#     cv2.imwrite("src/temp/registration/equalized_section_cropped.jpg", equalized)
#     uploaded_in_previous_step = Image.open("src/temp/registration/equalized_section_cropped.jpg")
#     uploaded_array = np.array(uploaded_in_previous_step)
#     img_height, img_width = uploaded_array.shape

#     moving_hem_path = "src/temp/registration/equalized_section_cropped.jpg"
#     fixed_hem_path =  "src/temp/registration/fixed_hem.jpg"

#     convert_image_nii(moving_hem_path,  "src/temp/registration/moving_hem.nii")
#     convert_image_nii(fixed_hem_path,   "src/temp/registration/fixed_hem.nii")

#     moving_nii_path = "src/temp/registration/moving_hem.nii"
#     fixed_nii_path =  "src/temp/registration/fixed_hem.nii"

#     resampled_moving = dpi_fixing(moving_nii_path,fixed_nii_path)
#     resampled_fixed = dpi_fixing(fixed_nii_path,fixed_nii_path)

#     sitk.WriteImage(resampled_moving, "src/temp/registration/resampled_moving_hem.nii" )
#     sitk.WriteImage(resampled_fixed,  "src/temp/registration/resampled_fixed_hem.nii")

#     resampled_moving_path = "src/temp/registration/resampled_moving_hem.nii"
#     resampled_fixed_path =  "src/temp/registration/resampled_fixed_hem.nii"

#     rigid_result =          rigid(resampled_fixed_path, resampled_moving_path)
#     affine_result =        affine(resampled_fixed_path, resampled_moving_path)
#     non_rigid_result =  non_rigid(resampled_fixed_path, resampled_moving_path)

#     convert_to_jpg(rigid_result, "src/temp/registration/rigid_hem.jpg")
#     convert_to_jpg(affine_result, "src/temp/registration/affine_hem.jpg" )
#     convert_to_jpg(non_rigid_result, "src/temp/registration/non_rigid_hem.jpg")

#     st.image([Image.open("src/temp/registration/rigid_hem.jpg"), 
#               Image.open("src/temp/registration/affine_hem.jpg"), 
#               Image.open("src/temp/registration/non_rigid_hem.jpg")], width=200, caption=["Rigid Registration", "Affine Registration", "Non-rigid Registration"])




# reg_results()
# # hem_reg_results()


# def rotate_image(image):
#     image= Image.open(image)
#     image_rot = image.rotate(-90)
#     flipped = np.fliplr(np.array(image_rot))
#     image_rot = Image.fromarray(flipped)
#     return image_rot

# rotated_image = rotate_image("src/temp/registration/non_rigid.jpg")
# rotated_image.save("src/temp/registration/non_rigid_rot.jpg")