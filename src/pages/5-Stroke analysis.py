import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
from module.utils import post_process_mask, smooth_mask, save_array_as_image, create_superpixel_image
from module.detection import seg_anything
from module.utils import equalize_this, resize_image_aspect_ratio, translate_bbox_to_original, get_bounding_box, create_binary_mask
from module.alignment import alignment, is_aligned
from streamlit_extras.switch_page_button import switch_page

st.header("Stroke Lesion Segmentation")
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
col1, divider, col2 = st.columns([4, 0.1, 6])

# with col1:
#     instruction_image_path = "raw_data/stroke_instructions.png"  # Replace with the path to your instruction image
#     instruction_image = Image.open(instruction_image_path)
#     st.write('Example usage. Follow the instructions!')
#     st.image(instruction_image)  # You can remove the caption if not needed

# with divider:
#     # This creates a thin, tall column that acts as a divider
#     st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
#     st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)
    
# with col2:
#     st.write('Draw bounding box in the following image.')

#     resized_image, scaling_factor = resize_image_aspect_ratio('results/alignment/equalized_aligned_section.jpg', 'results/alignment/equalized_aligned_section1.jpg')
#     uploaded_in_previous_step = Image.open("results/alignment/equalized_aligned_section1.jpg")


#     # sp = create_superpixel_image("src/temp/alignment/equalized_aligned_section.jpg")
#     # # equalized_array = sp.astype('uint8')
#     # equalized_pil_image = Image.fromarray(equalized_array)
#     # equalized_pil_image.save("src/temp/alignment/sp.jpg")

#     uploaded_array = np.array(uploaded_in_previous_step)

#     img_height, img_width=    uploaded_array.shape
#     max_canvas_width = 400
#     max_canvas_height = 400
#     scale_factor = min(max_canvas_width / img_width, 1)  # Ensuring the scale factor is not more than 1
#     scaled_width = int(img_width * scale_factor)
#     scaled_height = int(img_height * scale_factor)

#     drawing_mode = st.sidebar.selectbox(
#         "Drawing tool:", ("rect", "circle", "transform", "polygon")
#     )
#     stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
#     if drawing_mode == 'point':
#         point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
#     stroke_color = st.sidebar.color_picker("Stroke color hex: ")
#     bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

#     realtime_update = st.sidebar.checkbox("Update in realtime", True)
#     # Open the uploaded image with PIL
#     if uploaded_in_previous_step:
#         # Draw the canvas
#         canvas_result = st_canvas(fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#             stroke_width=stroke_width,
#             stroke_color=stroke_color,
#             background_color=bg_color,
#             background_image=uploaded_in_previous_step,
#             update_streamlit=realtime_update,
#             height=img_height,
#             width=img_width,
#             drawing_mode=drawing_mode,
#             key="canvas",
#         )
#         # if canvas_result.image_data is not None:
#         #     st.image(canvas_result.image_data)
#         process_button = st.button("Process Image")

#         objects = canvas_result.json_data['objects']
#         bbox_array = np.array([[obj['left'], obj['top'], obj['width'], obj['height']] for obj in objects])

#         if len(bbox_array) > 0:   
#             bbox_coords = {'x': bbox_array[0][0], 'y': bbox_array[0][1], 'width': bbox_array[0][2], 'height': bbox_array[0][3]}
#             print(bbox_coords)
#             bbox_coords = translate_bbox_to_original(bbox_coords, scaling_factor)

#             # seg_results = seg_anything("src/temp/registration/non_rigid_rot.jpg", bbox_coords)
#             if process_button:
#                 seg_results = seg_anything("results/alignment/equalized_aligned_section.jpg", bbox_coords)

#                 if seg_results:
#                     image_hem1 = Image.open('results/detection/source_image_hem1.jpg')
#                     seg_hem1 = Image.open('results/detection/segmented_image_hem1.jpg')
                    
#                     mask1_hem1 = post_process_mask('results/detection/mask1_hem1.jpg')
#                     mask2_hem1 = post_process_mask('results/detection/mask2_hem1.jpg')
#                     mask3_hem1 = post_process_mask("results/detection/mask3_hem1.jpg")
#                     save_array_as_image(mask1_hem1, 'results/detection/mask1_hem1.jpg' )
#                     save_array_as_image(mask2_hem1, 'results/detection/mask2_hem1.jpg' )
#                     save_array_as_image(mask3_hem1, 'results/detection/mask3_hem1.jpg' )

#                     st.image([image_hem1, seg_hem1], width=300, caption=["Hemisphere 1", "Segmented Image - Hemisphere 1"])
#                     st.image([mask2_hem1, mask3_hem1], width=300, caption=["1st  possible stroke mask", "2nd  possible stroke mask"])

#                     image_hem2 = Image.open('results/detection/source_image_hem2.jpg')
#                     seg_hem2 = Image.open('results/detection/segmented_image_hem2.jpg')
                    
#                     mask1_hem2 = post_process_mask('results/detection/mask1_hem2.jpg')
#                     save_array_as_image(mask1_hem2, 'results/detection/mask1_hem2.jpg' )

#                     st.image([image_hem2, seg_hem2], width=300, caption=["Hemisphere 2", "Segmented Image - Hemisphere 2"])
            
#                     # delete_foldercontents("results")
#         else:
#             st.warning('No bounding box drawn. Please draw a bounding box to proceed.')
            

with col1:
    st.subheader("Given section")
    resized_image, scaling_factor = resize_image_aspect_ratio('results/alignment/equalized_aligned_section.jpg', 'results/alignment/equalized_aligned_section1.jpg')
    # uploaded_in_previous_step = Image.open("results/alignment/equalized_aligned_section1.jpg")
    st.image("results/alignment/equalized_aligned_section1.jpg")


with col2:
    st.subheader("Results: stroke detection + masks")
    
    create_binary_mask('results/alignment/aligned_left_hemisphere.jpg', 'results/alignment/mask_left_hemisphere.jpg' )
    create_binary_mask('results/alignment/aligned_right_hemisphere.jpg', 'results/alignment/mask_right_hemisphere.jpg')

    left_hem = Image.open("results/alignment/mask_left_hemisphere.jpg")
    left_hem_array = np.array(left_hem)
    processed=post_process_mask('results/alignment/aligned_left_hemisphere.jpg')
    bbox = get_bounding_box(processed)
    # print(bbox)

    bbox_coords = {'x': bbox[0], 'y': bbox[1], 'width': bbox[2] - bbox[0], 'height': bbox[3] - bbox[1]}
    # bbox_coords = translate_bbox_to_original(bbox_coords, scaling_factor)

    seg_results = seg_anything("results/alignment/equalized_aligned_section.jpg", bbox_coords)
    if seg_results:
        image_hem1 = Image.open('results/detection/source_image_hem1.jpg')
        seg_hem1 = Image.open('results/detection/segmented_image_hem1.jpg')
        
        mask1_hem1 = post_process_mask('results/detection/mask1_hem1.jpg')
        mask2_hem1 = post_process_mask('results/detection/mask2_hem1.jpg')
        mask3_hem1 = post_process_mask("results/detection/mask3_hem1.jpg")
        save_array_as_image(mask1_hem1, 'results/detection/mask1_hem1.jpg')
        save_array_as_image(mask2_hem1, 'results/detection/mask2_hem1.jpg')
        save_array_as_image(mask3_hem1, 'results/detection/mask3_hem1.jpg')

        # st.image([seg_hem1], width=300, caption=["Segmented Image - Hemisphere 1"])
        st.image([mask2_hem1, mask3_hem1], width=300, caption=["1st possible stroke mask", "2nd possible stroke mask"])

        # image_hem2 = Image.open('results/detection/source_image_hem2.jpg')
        seg_hem2 = Image.open('results/detection/segmented_image_hem2.jpg')
        
        mask1_hem2 = post_process_mask('results/detection/mask1_hem2.jpg')
        save_array_as_image(mask1_hem2, 'results/detection/mask1_hem2.jpg')

        # st.image([seg_hem2], width=300, caption=["Segmented Image - Hemisphere 2"])

        left_part_image = Image.open('results/alignment/aligned_left_hemisphere.jpg')
        right_part_image = Image.open('results/alignment/aligned_right_hemisphere.jpg')

        st.image([left_part_image, right_part_image], width=300, caption=["Left Hemisphere", "Right Hemisphere"])


col1, col2, col3 = st.columns([1,1,1])
with col2: 
    if st.button("Next"):
        switch_page("anatomical mapping")
