import streamlit as st
from PIL import Image
import io
from streamlit_image_select import image_select
import numpy as np
import os

st.header("Mapping Section to Atlas")

def rotate_image(image, degrees):
    """
    Rotates the image by the specified number of degrees
    """
    file_bytes = io.BytesIO(image.read())
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
    uploaded_file = st.file_uploader("Upload a section", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_file:
        image = st.image(uploaded_file,width=300)
        image = Image.open(uploaded_file).convert("RGB")
        image.save("src/temp/mapping/uploaded_section.jpg")

with col2:
    folder_path = 'raw_data/atlas/'
    photos = read_photos_from_folder(folder_path)

    if len(photos) > 0:
        carousel_index = st.slider("Choose corresponding atlas section", 1, len(photos)-1, 1)
        selected_image_path = photos[carousel_index]
        image = st.image(selected_image_path, use_column_width=True)

        # Button for confirming the selection of the currently displayed image
        if st.button('Select this Atlas Section'):
            # For example, saving the selected image path or performing further operations
            selected = Image.open(selected_image_path).convert("RGB")
            selected.save("src/temp/mapping/mapped_allen_section.jpg")
            st.success(f"Selected {os.path.basename(selected_image_path)} as the atlas section.")
    else:
        st.warning("No photos found in the specified folder.")