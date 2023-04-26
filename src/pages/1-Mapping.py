import streamlit as st
from PIL import Image
import io
from streamlit_image_select import image_select
import numpy as np


"""
# Mapping with Atlas

"""
import os

def rotate_image(image, degrees):
    """
    Rotates the image by the specified number of degrees
    """
    file_bytes = io.BytesIO(file.read())
    image = Image.open(file_bytes)

    rotated_image = image.rotate(degrees)
    return rotated_image


def read_photos_from_folder(folder_path):
    """Reads all photos from a local folder."""
    photos = []
    contents = os.listdir(folder_path)
    # Sort the contents alphabetically
    contents.sort()
    for file_name in contents:
        if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
            photo_path = os.path.join(folder_path, file_name)
            photos.append(photo_path)
            # st.write(file_name)
    return photos


col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Upload a section", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    for file in uploaded_files:
        if file:
            image = st.image(file,width=300)

            rotation_degrees = st.number_input('Degrees to rotate the image:', min_value=-360, max_value=360, value=0, step=1)
            st.write('The current number is ', rotation_degrees)
            rotated_image = rotate_image(image, rotation_degrees)

            # Show rotated image
            st.image(rotated_image, caption="Rotated Image")
               


              
with col2:
    folder_path = '/home/ioanna/Documents/Thesis/data/atlas'

    # Read photos from the local folder
    photos = read_photos_from_folder(folder_path)

    # Display the photos in a carousel using Streamlit
    if len(photos) > 0:
        # carousel_index = st.slider("Choose corresponding atlas section", 0, len(photos)-1, 1)
        # image = st.image(photos[carousel_index], use_column_width=True)
        
        # if "rotate_slider" not in st.session_state:
        #     st.session_state["rotate_slider"] = 0
        # rotation_degrees = st.number_input('Degrees to rotate the image:', min_value=-360, max_value=360, value=st.session_state["rotate_slider"], key="rotate_slider")
        img = image_select(
            label="Select corresponding atlas section",
            images=photos,
            
        )
        st.write(str(img))
    else:
        st.warning("No photos found in the specified folder.")



