import streamlit as st
from PIL import Image
import io
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
    return photos

st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        flex-direction: row;
    }
    .vertical-line {
        border-left: 2px solid #000; /* Black color line */
        height: auto;
        position: absolute;
        left: 50%;
        margin-left: -1px;
        top: 0;
        bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize the session state for the carousel index
if 'carousel_index' not in st.session_state:
    st.session_state.carousel_index = 0

# col1, col2 = st.columns(2)
col1, divider, col2 = st.columns([5, 0.1, 5])
with col1:
    uploaded_file = st.file_uploader("Upload a section using browse files or use drag and drop.", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = st.image(uploaded_file, width=300)
        image = Image.open(uploaded_file).convert("RGB")
        image.save("results/mapping/uploaded_section.jpg")

with divider:
    # This creates a thin, tall column that acts as a divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)

with col2:
    folder_path = 'raw_data/atlas1/'
    photos = read_photos_from_folder(folder_path)

    if len(photos) > 0:
        # Display buttons for previous and next side by side
        st.write('Use the "Previous" and "Next" buttons for fine navigation, or the slider for larger jumps through the atlas sections.')
        col_prev, col_next = st.columns(2)
    
        if col_prev.button('Previous', key='prev'):
            if st.session_state.carousel_index > 0:
                st.session_state.carousel_index -= 1
                # Update the slider to reflect the change from the button press
                st.session_state.slider_index = st.session_state.carousel_index
        
        if col_next.button('Next', key='next'):
            if st.session_state.carousel_index < len(photos) - 1:
                st.session_state.carousel_index += 1
                # Update the slider to reflect the change from the button press
                st.session_state.slider_index = st.session_state.carousel_index
        
        # Create a slider for jumping to specific images
        # Initialize the slider index in session state if not present
        if 'slider_index' not in st.session_state:
            st.session_state.slider_index = 0

        slider_index = st.slider(
            "Choose corresponding atlas section", 
            0, 
            len(photos) - 1, 
            st.session_state.slider_index
        )
        
        # Update the carousel index if the slider is used
        if slider_index != st.session_state.carousel_index:
            st.session_state.carousel_index = slider_index

        # Show the image at the current index
        selected_image_path = photos[st.session_state.carousel_index]
        st.image(selected_image_path, use_column_width=True)

        # Button for confirming the selection of the currently displayed image
        if st.button('Select this Atlas Section', key='select'):
            selected = Image.open(selected_image_path).convert("RGB")
            selected.save("results/mapping/mapped_allen_section.jpg")
            st.success(f"Selected {os.path.basename(selected_image_path)} as the atlas section.")
    else:
        st.warning("No photos found in the specified folder.")