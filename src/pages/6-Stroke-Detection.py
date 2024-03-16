import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
from module.utils import post_process_mask, smooth_mask, save_array_as_image, create_superpixel_image
from module.detection import seg_anything
from module.utils import equalize_this
from module.alignment import alignment, is_aligned


st.header("Stroke Lesion Detection and obtained masks visualization")
col1,col2 = st.columns(2)

uploaded_in_previous_step = Image.open("results/alignment/equalized_aligned_section.jpg")
# sp = create_superpixel_image("src/temp/alignment/equalized_aligned_section.jpg")
# # equalized_array = sp.astype('uint8')
# equalized_pil_image = Image.fromarray(equalized_array)
# equalized_pil_image.save("src/temp/alignment/sp.jpg")


uploaded_array = np.array(uploaded_in_previous_step)
img_height, img_width= uploaded_array.shape


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
        # seg_results = seg_anything("src/temp/registration/non_rigid_rot.jpg", bbox_coords)

        seg_results = seg_anything("results/alignment/equalized_aligned_section.jpg", bbox_coords)

        if seg_results:
            image_hem1 = Image.open('results/detection/source_image_hem1.jpg')
            seg_hem1 = Image.open('results/detection/segmented_image_hem1.jpg')

            mask1_hem1 = post_process_mask('results/detection/mask1_hem1.jpg')
            mask2_hem1 = post_process_mask('results/detection/mask2_hem1.jpg')
            mask3_hem1 = post_process_mask("results/detection/mask3_hem1.jpg")
            save_array_as_image(mask1_hem1, 'results/detection/mask1_hem1.jpg' )
            save_array_as_image(mask2_hem1, 'results/detection/mask2_hem1.jpg' )
            save_array_as_image(mask3_hem1, 'results/detection/mask3_hem1.jpg' )

                            

            st.image([image_hem1, seg_hem1], width=300)
            st.image([mask1_hem1, mask2_hem1, mask3_hem1], width=200)


            image_hem2 = Image.open('results/detection/source_image_hem2.jpg')
            seg_hem2 = Image.open('results/detection/segmented_image_hem2.jpg')
            
            mask1_hem2 = post_process_mask('results/detection/mask1_hem2.jpg')
            save_array_as_image(mask1_hem2, 'results/detection/mask1_hem2.jpg' )
    

            st.image([image_hem2, seg_hem2], width=300)
            # st.image([mask1_hem2, mask2_hem2, mask3_hem2], width=200)
            st.image([mask1_hem2], width=200)


            # delete_foldercontents("results")
    else:
        st.warning('No bounding box drawn. Please draw a bounding box to proceed.')
        