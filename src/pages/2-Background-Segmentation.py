import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
from module.detection import seg_anything_bgs
from module.utils import equalize_this
import os, shutil
from module.utils import post_process_mask, smooth_mask, save_array_as_image, create_superpixel_image
import cv2

def save_uploadedfile(uploadedfile, path):
    with open(os.path.join(path),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved uploaded image to a temporary folder")

def delete_foldercontents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


from PIL import Image

def resize_image_aspect_ratio(image_path, output_path, max_size=400):
    # Open the original image
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    # Calculate the scaling factor
    scaling_factor = max_size / max(original_width, original_height)
    
    # Calculate the new dimensions
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save the resized image
    resized_image.save(output_path)

    print(f"Image has been resized to {new_width}x{new_height} and saved successfully!")
    
    # Return the resized image and the scaling factors
    return resized_image, scaling_factor# Example usage

def translate_bbox_to_original(bbox_resized, scaling_factor):
    """
    Translate bounding box coordinates from the resized dimensions back to the original image dimensions.

    Parameters:
    - bbox_resized: The bounding box in the resized image, as a dictionary with keys 'x', 'y', 'width', 'height'.
    - scaling_factor: The scaling factor used to resize the image.

    Returns:
    - A dictionary containing the bounding box coordinates translated back to the original image dimensions.
    """
    
    # Translate the bounding box coordinates
    bbox_original = {
        'x': bbox_resized['x'] / scaling_factor,
        'y': bbox_resized['y'] / scaling_factor,
        'width': bbox_resized['width'] / scaling_factor,
        'height': bbox_resized['height'] / scaling_factor
    }
    
    return bbox_original



################################################################################################################################################
st.header("Background Segmentation")

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

col1, divider, col2 = st.columns([3, 0.1, 7 ])

with col1:
    instruction_image_path = "raw_data/instructions.png"  # Replace with the path to your instruction image
    instruction_image = Image.open(instruction_image_path)
    st.write('Example usage. Follow the instructions!')

    st.image(instruction_image) 

with divider:
    # This creates a thin, tall column that acts as a divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)

with col2:
    st.write('Draw bounding box in the following image.')

    resized_image, scaling_factor = resize_image_aspect_ratio('results/mapping/uploaded_section.jpg', 'results/mapping/uploaded_section1.jpg')
    uploaded_in_previous_step = Image.open("results/mapping/uploaded_section1.jpg")

    
    uploaded_array = np.array(uploaded_in_previous_step)

    img_height, img_width, _=    uploaded_array.shape
    max_canvas_width = 400
    max_canvas_height = 400
    # Calculate scaled dimensions
    scale_factor = min(max_canvas_width / img_width, 1)  # Ensuring the scale factor is not more than 1
    scaled_width = int(img_width * scale_factor)
    scaled_height = int(img_height * scale_factor)

    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("rect", "circle", "transform", "polygon")
    )

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    if uploaded_in_previous_step:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)", 
            stroke_width=stroke_width, 
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=uploaded_in_previous_step.resize((scaled_width, scaled_height)),
            update_streamlit=True,
            height=scaled_height,
            width=scaled_width,
            drawing_mode=drawing_mode,  
            key="canvas",
        )

        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data, use_column_width=True)

        objects = canvas_result.json_data['objects']
        bbox_array = np.array([[obj['left'], obj['top'], obj['width'], obj['height']] for obj in objects])


        if len(bbox_array) > 0:   
            bbox_coords = {'x': bbox_array[0][0], 'y': bbox_array[0][1], 'width': bbox_array[0][2], 'height': bbox_array[0][3]}
            print(bbox_coords)
            bbox_coords = translate_bbox_to_original(bbox_coords, scaling_factor)
            seg_results = seg_anything_bgs("results/mapping/uploaded_section.jpg", bbox_coords)

            if seg_results:
                seg_image = Image.open('results/segmentation/segmented_image.jpg')
                seg_mask = Image.open('results/segmentation/mask_bgs.jpg')
                seg_mask2 = Image.open('results/segmentation/mask2_bgs.jpg')


                
                seg_mask = post_process_mask('results/segmentation/mask_bgs.jpg')
                seg_mask2 = post_process_mask('results/segmentation/mask2_bgs.jpg')
                save_array_as_image(seg_mask, 'results/segmentation/mask_bgs.jpg' )
                save_array_as_image(seg_mask2, 'results/segmentation/mask2_bgs.jpg' )

            
                st.image([seg_image, seg_mask, seg_mask2], width=300, caption=["Segmented image", "1st  possible BGS mask", "2nd  possible BGS mask"])
        else:
            st.warning('No bounding box drawn. Please draw a bounding box to proceed.')





