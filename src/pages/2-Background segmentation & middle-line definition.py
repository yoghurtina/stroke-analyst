import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
from module.detection import seg_anything_bgs
from module.utils import equalize_this
import os, shutil
from module.utils import post_process_mask, smooth_mask, save_array_as_image, create_superpixel_image, create_binary_mask
import cv2
from module.utils import get_segmented, translate_bbox_to_original, resize_image_aspect_ratio, create_mask_from_path, split_image, get_segmented_hemispheres
from streamlit_extras.switch_page_button import switch_page

################################################################################################################################################
st.header("Background segmentation and middle-line definition")

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
if 'seg_image' not in st.session_state:
    st.session_state['seg_image'] = None

if 'bbox' not in st.session_state:
    st.session_state['bbox'] = None

if 'path' not in st.session_state:
    st.session_state['path'] = None

col1, divider, col2, divider2, col3 = st.columns([2, 0.1, 4, 0.1, 4])

with col1:
    st.subheader("Example usage. Follow the instructions")

    instruction_image_path = "raw_data/instructions.png"  # Replace with the path to your instruction image
    instruction_image = Image.open(instruction_image_path)
    st.write('Example usage. Follow the instructions!')

    st.image(instruction_image) 

with divider:

    # This creates a thin, tall column that acts as a divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)


with col2:
    st.subheader("Segment background and define middle line")

    st.write('Draw bounding box as close as possible to the section. Draw the middle line of the section. Once ready, press Process button.')
    resized_image, scaling_factor = resize_image_aspect_ratio('results/mapping/uploaded_section.jpg', 'results/mapping/uploaded_section1.jpg')
    uploaded_section = Image.open("results/mapping/uploaded_section1.jpg")
    uploaded_array = np.array(uploaded_section)
    

    drawing_mode = st.sidebar.radio("Choose drawing mode:", ("Bounding Box", "Free Draw"))

    img_height, img_width, _=    uploaded_array.shape
    max_canvas_width = 300
    max_canvas_height = 300
    # Calculate scaled dimensions
    scale_factor = min(max_canvas_width / img_width, 1)  # Ensuring the scale factor is not more than 1
    scaled_width = int(img_width * scale_factor)
    scaled_height = int(img_height * scale_factor)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="default",
        background_color="default",
        background_image=uploaded_section.resize((scaled_width, scaled_height)),
        update_streamlit=True,
        height=scaled_height, 
        width=scaled_width,  
        drawing_mode="freedraw" if drawing_mode == "Free Draw" else "rect",
        key="canvas"
    )
    process_button = st.button("Process Image")
    if process_button:
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data['objects']
            for obj in objects:
                if obj['type'] == 'rect':
                    bbox_resized = {'x': obj['left'], 'y': obj['top'], 'width': obj['width'], 'height': obj['height']}
                    st.session_state['bbox'] = translate_bbox_to_original(bbox_resized, scaling_factor)
                elif obj['type'] == 'path':
                    st.session_state['path'] = [segment for segment in obj['path']]
        else:
            st.warning("Please draw a bounding box and the section's middle line before processing.")

with divider2:
    # This creates a thin, tall column that acts as a divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)

with col3:
    st.subheader("Results: segmented section + masks")
    if st.session_state['bbox']:
        seg_results = seg_anything_bgs("results/mapping/uploaded_section.jpg", st.session_state['bbox'])
        if seg_results:
            st.session_state['seg_image'] = Image.open('results/segmentation/segmented_image.jpg')

            seg_image = Image.open('results/segmentation/segmented_image.jpg')
            seg_mask = Image.open('results/segmentation/mask_bgs.jpg')
            # seg_mask2 = Image.open('results/segmentation/mask2_bgs.jpg')

            seg_mask = post_process_mask('results/segmentation/mask_bgs.jpg')
            # seg_mask2 = post_process_mask('results/segmentation/mask2_bgs.jpg')
            save_array_as_image(seg_mask, 'results/segmentation/mask_bgs.jpg')
            # save_array_as_image(seg_mask2, 'results/segmentation/mask2_bgs.jpg')

            st.image([seg_image, seg_mask], width=300, caption=["Segmented section", "BGS mask"])

    if st.session_state['seg_image'] is not None and st.session_state['path']:
        segmented_image = get_segmented_hemispheres('results/segmentation/segmented_image.jpg', 'results/segmentation/mask_bgs.jpg')

        seg_image_array = np.array(segmented_image)  # Convert PIL Image to numpy array
        mask = create_mask_from_path(st.session_state['path'], segmented_image.shape, scaling_factor)
        left_part, right_part = split_image(seg_image_array, mask)
        left_part_image = Image.fromarray(left_part)
        right_part_image = Image.fromarray(right_part)

        left_part_image.save('results/segmentation/left_part.jpg')
        right_part_image.save('results/segmentation/right_part.jpg')
        create_binary_mask('results/segmentation/left_part.jpg', 'results/segmentation/mask_left_part.jpg' )
        create_binary_mask('results/segmentation/right_part.jpg', 'results/segmentation/mask_right_part.jpg')
        st.image([left_part_image, right_part_image], width=300, caption=["Left Hemisphere", "Right Hemisphere"])


col1, col2, col3 = st.columns([1,1,1])
with col2: 
    if st.button("Next"):
        switch_page("rotation correction & registration")
