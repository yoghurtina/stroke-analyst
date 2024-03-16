import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
from module.detection import seg_anything, seg_anything_bgs
from module.utils import equalize_this
import os, shutil
from io import BytesIO
import cv2

st.header("Background Segmentation")
col1,col2 = st.columns(2)

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


uploaded_in_previous_step = Image.open("results/mapping/uploaded_section.jpg")
uploaded_array = np.array(uploaded_in_previous_step)
img_height, img_width, _= uploaded_array.shape


drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("rect", "circle", "transform", "polygon")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Open the uploaded image with PIL
if uploaded_in_previous_step:
    # Draw the canvas
    canvas_result = st_canvas(fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=uploaded_in_previous_step,
        update_streamlit=realtime_update,
        height=img_height,
        width=img_width,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)

    objects = canvas_result.json_data['objects']
    bbox_array = np.array([[obj['left'], obj['top'], obj['width'], obj['height']] for obj in objects])


    if len(bbox_array) > 0:   
        bbox_coords = {'x': bbox_array[0][0], 'y': bbox_array[0][1], 'width': bbox_array[0][2], 'height': bbox_array[0][3]}
        print(bbox_coords)

        seg_results = seg_anything_bgs("results/mapping/uploaded_section.jpg", bbox_coords)

        if seg_results:
            seg_image = Image.open('results/segmentation/segmented_image.jpg')
            seg_mask = Image.open('results/segmentation/mask_bgs.jpg')
        
            st.image([seg_image, seg_mask], width=300)
            # delete_foldercontents("/home/ioanna/Documents/Thesis/src/temp")
    else:
        st.warning('No bounding box drawn. Please draw a bounding box to proceed.')

