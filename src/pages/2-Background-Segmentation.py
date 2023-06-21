import numpy as np
import streamlit as st
from module.segmentation import segmentation
from io import BytesIO
from pathlib import Path
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.header("Background Segmentation of the section")

def load_image(image_file):
    img = Image.open(image_file)
    return img


def save_uploadedfile(uploadedfile):
    with open(os.path.join("temp/segmentation","background_segmented_section.jpg"),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{}".format("background_segmented_section"))

uploaded_in_previous_step = Image.open("src/temp/mapping/uploaded_section.jpg")
selected_allen_section =    Image.open("src/temp/mapping/mapped_allen_section.jpg") 

st.image([uploaded_in_previous_step, selected_allen_section], width=300, caption=["Previously Uploaded Section", "Previously Mapped Allen Section"])

def main():
    uploaded_file = st.file_uploader("Upload one mouse brain section", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    # uploaded_file = Image.open("/home/ioanna/Documents/Thesis/src/temp/mapping/uploaded_section.jpg")

    # uploaded_file = uploaded_in_previous_step
    if uploaded_file:
        # image = Image.open(uploaded_file)

        image = st.image(uploaded_file,width=300, caption="Uploaded Section")
            # Initialization
        if 'button' not in st.session_state:
            st.session_state['button'] = 'value'

        if st.button('Segment image', key = "value"):

            seg, mask = segmentation(uploaded_file)
            segmentated = st.image(seg,width=300, caption="Background Segmented Section")
            mask_ = st.image(mask,width=300, caption="Background Segmented Mask")
            
            # img = load_image(uploaded_file)
            # save_uploadedfile(uploaded_file)
            segmented_pil_image = Image.fromarray(seg)
            mask_pil_image = Image.fromarray(mask)
            segmented_pil_image.save("src/temp/segmentation/background_segmented_image.jpg")
            mask_pil_image.save("src/temp/segmentation/mask_segmented_image.jpg")

if __name__ == "__main__":
    main()