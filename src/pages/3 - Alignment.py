import cv2
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
from module.segmentation import segmentation
from module.alignment import alignment, is_aligned
from PIL import Image
import io

def rotate_image(image, degrees):
    """
    Rotates the image by the specified number of degrees
    """
    rotated_image = image.rotate(degrees)
    return rotated_image

st.title("Image Alignment")

def main():
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    for file in uploaded_files:
        if file:
            file_bytes = io.BytesIO(file.read())
            image = Image.open(file_bytes)

            st.image(image, caption="original image")

            # if st.button('Align image'):
            # for file in uploaded_files:
            aligned = alignment(file)
            aligned = Image.fromarray(aligned)
            # frame = np.array(image)
            st.image(aligned, caption="rotated image")

            rotation_degrees = st.number_input('Degrees to rotate the image:', min_value=-360, max_value=360, value=0, step=1)
            st.write('The current number is ', rotation_degrees)
            rotated_image = rotate_image(aligned, rotation_degrees)

                    # Show rotated image
            st.image(rotated_image, caption="Rotated Image")
                # download_button = st.download_button(
                #                 label="Download segmentated image",
                #                 data=segmentated,
                #                 file_name="test",
                #                 mime="image/png"
                #             )
              
              

    
if __name__ == "__main__":
    main()





# # Upload image
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read image data into memory
#     file_bytes = io.BytesIO(uploaded_file.read())

#     # Load image
#     image = Image.open(file_bytes)

#     # Show original image
#     st.image(image, caption="Original Image")
#     # Rotate image
#     rotation_degrees = st.slider("Rotate by how many degrees?", -360, 360, 0)
#     rotated_image = rotate_image(image, rotation_degrees)

#     # Show rotated image
#     st.image(rotated_image, caption="Rotated Image")
