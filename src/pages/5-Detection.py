import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
# from module.detection import seg_anything

col1,col2 = st.columns(2)

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("rect", "circle", "transform", "polygon")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

realtime_update = st.sidebar.checkbox("Update in realtime", True)



uploaded_file = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

# Load and display image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image', use_column_width=True)

# Open the uploaded image with PIL
if uploaded_file is not None:
    image = Image.open(uploaded_file)

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
        background_image=Image.open(uploaded_file) if uploaded_file else None,
        update_streamlit=realtime_update,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        point_display_radius=0,
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

# def hemisphere(bbox):
    # seg_anything(image, bbox)
    # pass
    
# hemisphere(bbox_coords)

