import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
from module.detection import seg_anything
from module.normalization import equalize_this
import os, shutil


col1,col2 = st.columns(2)

def save_uploadedfile(uploadedfile):
    with open(os.path.join("temp",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved uploaded image to a temporary folder".format(uploadedfile.name))


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



bg_image1 = st.sidebar.file_uploader("Background image:", type=["png", "jpg", "nii"])

# Open the uploaded image with PIL
if bg_image1 is not None:
    image = Image.open(bg_image1)

    # Get the width and height of the image
    img_width, img_height = image.size

    st.write(img_width, img_height)

    # Set the width and height of the canvas
    canvas_width = img_width
    canvas_height = img_height

    # Draw the canvas
    canvas_result = st_canvas(fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image1) if bg_image1 else None,
        update_streamlit=realtime_update,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    # Do something interesting with the image data and paths
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


            if bg_image1 is not None:

                file_details = {"FileName":bg_image1.name,"FileType":bg_image1.type}

                save_uploadedfile(bg_image1)
                # image=equalize_this("/home/ioanna/Documents/Thesis/src/temp/moving.jpg")

                re = seg_anything("/home/ioanna/Documents/Thesis/src/temp/moving.jpg", bbox_coords)

                if re:
                    image1 = Image.open('/home/ioanna/Documents/Thesis/src/temp/source_image.jpg')
                    image2 = Image.open('/home/ioanna/Documents/Thesis/src/temp/segmented_image.jpg')
                    
                    mask1 = Image.open('/home/ioanna/Documents/Thesis/src/temp/mask11.jpg')
                    mask2 = Image.open('/home/ioanna/Documents/Thesis/src/temp/mask12.jpg')
                    mask3 = Image.open('/home/ioanna/Documents/Thesis/src/temp/mask13.jpg')


                    st.image([image1, image2], width=300)
                    st.image([mask1, mask2, mask3], width=200)

        
                    # delete_foldercontents("/home/ioanna/Documents/Thesis/src/temp")
