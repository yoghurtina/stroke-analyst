import streamlit as st
from streamlit_cropper import st_cropper
import os
import numpy as np
from PIL import Image
from io import BytesIO
import SimpleITK as sitk

from module.reg_preprocessing import convert_image_nii, dpi_fixing, convert_to_grayscale, convert_to_jpg
from module.anatomy import rigid, non_rigid, affine


col1, col2 = st.columns(2)

def save_uploadedfile(uploadedfile, path):
    with open(os.path.join(path),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return True


realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "Free": None,
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3)
}
aspect_ratio = aspect_dict[aspect_choice]


def cropper():
    uploaded_in_previous_step = Image.open("src/temp/detection/equalized_section.jpg")

    
    if not realtime_update:
        st.write("Double click to save crop")

    cropped_img = st_cropper(uploaded_in_previous_step, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    st.write("Cropped Hemisphere")
    _ = cropped_img.thumbnail((200,200))
    st.image(cropped_img, use_column_width=300)
    # Convert the cropped image to bytes-like object
    byte_io = BytesIO()
    cropped_img.save(byte_io, format='JPEG')  # Adjust format as needed

    # Save the bytes-like object to a file
    save_uploadedfile(byte_io,"src/temp/anatomy/stroked_hemisphere.jpg")

def reg_results():

    uploaded_in_previous_step = Image.open("src/temp/detection/equalized_section.jpg")
    selected_atlas_section = Image.open("src/temp/mapping/mapped_allen_section.jpg")
    st.image([uploaded_in_previous_step, selected_atlas_section], caption=["Moving Image", "Fixed Image"], width=300)

    moving_image_path = "src/temp/detection/equalized_section.jpg"
    fixed_image_path =  "src/temp/mapping/mapped_allen_section.jpg"

    convert_image_nii(moving_image_path, "src/temp/anatomy/moving.nii")
    convert_image_nii(fixed_image_path,  "src/temp/anatomy/fixed.nii")

    moving_nii_path = "src/temp/anatomy/moving.nii"
    fixed_nii_path =  "src/temp/anatomy/fixed.nii"

    resampled_moving = dpi_fixing(moving_nii_path)
    resampled_fixed = dpi_fixing(fixed_nii_path)

    sitk.WriteImage(resampled_moving, "src/temp/anatomy/resampled_moving.nii" )
    sitk.WriteImage(resampled_fixed,  "src/temp/anatomy/resampled_fixed.nii")

    resampled_moving_path = "src/temp/anatomy/resampled_moving.nii"
    resampled_fixed_path =  "src/temp/anatomy/resampled_fixed.nii"

    rigid_result =          rigid(resampled_fixed_path, resampled_moving_path)
    affine_result =        affine(resampled_fixed_path, resampled_moving_path)
    non_rigid_result =  non_rigid(resampled_fixed_path, resampled_moving_path)

    convert_to_jpg(rigid_result, "src/temp/anatomy/rigid.jpg")
    convert_to_jpg(affine_result, "src/temp/anatomy/affine.jpg" )
    convert_to_jpg(non_rigid_result, "src/temp/anatomy/non_rigid.jpg")

    st.image([Image.open("src/temp/anatomy/rigid.jpg"), 
            Image.open("src/temp/anatomy/affine.jpg"), 
            Image.open("src/temp/anatomy/non_rigid.jpg")], width=200, caption=["Rigid Registration", "Affine Registration", "Non-rigid Registration"])

# moving_hem_path = ""
# fixed_hem_path = ""

# convert_image_nii(moving_hem_path,  "/home/ioanna/Documents/Thesis/src/temp/anatomy/moving_hem.nii")
# convert_image_nii(fixed_hem_path,  "/home/ioanna/Documents/Thesis/src/temp/anatomy/fixed_hem.nii")


st.header("Registration Results (Whole section and stroked hemisphere)")

# with col2:
#     col2.subheader("Hemisphere Cropper and Registration")
#     cropper()

def hem_reg_results():
    moving_hem_path = "src/temp/anatomy/stroked_hemisphere.jpg"
    fixed_hem_path =  "src/temp/anatomy/atlas_hem.jpg"

    convert_image_nii(moving_hem_path,  "src/temp/anatomy/moving_hem.nii")
    convert_image_nii(fixed_hem_path,   "src/temp/anatomy/fixed_hem.nii")

    moving_nii_path = "src/temp/anatomy/moving_hem.nii"
    fixed_nii_path =  "src/temp/anatomy/fixed_hem.nii"

    resampled_moving = dpi_fixing(moving_nii_path)
    resampled_fixed = dpi_fixing(fixed_nii_path)

    sitk.WriteImage(resampled_moving, "src/temp/anatomy/resampled_moving_hem.nii" )
    sitk.WriteImage(resampled_fixed,  "src/temp/anatomy/resampled_fixed_hem.nii")

    resampled_moving_path = "src/temp/anatomy/resampled_moving_hem.nii"
    resampled_fixed_path =  "src/temp/anatomy/resampled_fixed_hem.nii"

    rigid_result =          rigid(resampled_fixed_path, resampled_moving_path)
    affine_result =        affine(resampled_fixed_path, resampled_moving_path)
    non_rigid_result =  non_rigid(resampled_fixed_path, resampled_moving_path)

    convert_to_jpg(rigid_result, "src/temp/anatomy/rigid_hem.jpg")
    convert_to_jpg(affine_result, "src/temp/anatomy/affine_hem.jpg" )
    convert_to_jpg(non_rigid_result, "src/temp/anatomy/non_rigid_hem.jpg")

    st.image([Image.open("src/temp/anatomy/rigid_hem.jpg"), 
              Image.open("src/temp/anatomy/affine_hem.jpg"), 
              Image.open("src/temp/anatomy/non_rigid_hem.jpg")], width=200, caption=["Rigid Registration", "Affine Registration", "Non-rigid Registration"])




reg_results()
hem_reg_results()