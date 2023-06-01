import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
from module.detection import seg_anything
from module.normalization import equalize_this
import os, shutil
from io import BytesIO


st.header("Stroke Lesion Detection")
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



uploaded_in_previous_step = Image.open("src/temp/alignment/aligned_section.jpg")
uploaded_array = np.array(uploaded_in_previous_step)
img_height, img_width, _ = uploaded_array.shape

# equalized = equalize_this("/home/ioanna/Documents/Thesis/src/temp/alignment/aligned_section.jpg")
# equalized_pil_img  = Image.fromarray(equalized)
# byte_io = BytesIO()
# equalized_pil_img.save(byte_io, format='JPEG')  # Adjust format as needed
# # Save the bytes-like object to a file
# save_uploadedfile(byte_io,"/home/ioanna/Documents/Thesis/src/temp/detection/equalized_section.jpg")
# print(type(uploaded_in_previous_step))


# print(img_height, img_width)
# image = st.image(uploaded_in_previous_step)
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("rect", "circle", "transform", "polygon")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

realtime_update = st.sidebar.checkbox("Update in realtime", True)


# bg_image1 = st.sidebar.file_uploader("Background image:", type=["png", "jpg", "jpeg"])

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

    if canvas_result.json_data is not None:
        # st.write((canvas_result.json_data['objects']))
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)

        objects = canvas_result.json_data['objects']
        bbox_array = np.array([[obj['left'], obj['top'], obj['width'], obj['height']] for obj in objects])
        print(bbox_array)


        if bbox_array is not None:
            bbox_coords = {'x': bbox_array[0][0], 'y': bbox_array[0][1], 'width': bbox_array[0][2], 'height': bbox_array[0][3]}
            print(bbox_coords)

            seg_results = seg_anything("src/temp/alignment/aligned_section.jpg", bbox_coords)

            if seg_results:
                image_hem1 = Image.open('src/temp/detection/source_image_hem1.jpg')
                seg_hem1 = Image.open('src/temp/detection/segmented_image_hem1.jpg')
                
                mask1_hem1 = Image.open('src/temp/detection/mask1_hem1.jpg')
                mask2_hem1 = Image.open('src/temp/detection/mask2_hem1.jpg')
                mask3_hem1 = Image.open('src/temp/detection/mask3_hem1.jpg')


                st.image([image_hem1, seg_hem1], width=300)
                st.image([mask1_hem1, mask2_hem1, mask3_hem1], width=200)


                image_hem2 = Image.open('src/temp/detection/source_image_hem2.jpg')
                seg_hem2 = Image.open('src/temp/detection/segmented_image_hem2.jpg')
                
                mask1_hem2 = Image.open('src/temp/detection/mask1_hem2.jpg')
                mask2_hem2 = Image.open('src/temp/detection/mask2_hem2.jpg')
                mask3_hem2 = Image.open('src/temp/detection/mask3_hem2.jpg')


                st.image([image_hem2, seg_hem2], width=300)
                st.image([mask1_hem2, mask2_hem2, mask3_hem2], width=200)

                # delete_foldercontents("/home/ioanna/Documents/Thesis/src/temp")
