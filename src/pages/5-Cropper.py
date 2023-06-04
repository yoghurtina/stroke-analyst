import streamlit as st
from streamlit_cropper import st_cropper
import os
import numpy as np
from PIL import Image
from io import BytesIO
import SimpleITK as sitk

from module.reg_preprocessing import convert_image_nii, dpi_fixing, convert_to_grayscale, convert_to_jpg
from module.registration import rigid, non_rigid, affine


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
    st.image(cropped_img, use_column_width=200)
    # Convert the cropped image to bytes-like object
    byte_io = BytesIO()
    cropped_img.save(byte_io, format='JPEG')  # Adjust format as needed

    # Save the bytes-like object to a file
    save_uploadedfile(byte_io,"src/temp/registration/stroked_hemisphere.jpg")


cropper()