import cv2
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
from module.segmentation import segmentation
from module.alignment import alignment, is_aligned
from PIL import Image


st.title("Image Alignment")


def main():
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    for file in uploaded_files:
        if file:
            image = st.image(file)
            if st.button('Align image', key="1"):
            # for file in uploaded_files:
                aligned = alignment(file)
                # frame = np.array(image)
                aligned = st.image(aligned)
                
                degrees = st.slider('Rotate between 0 and 360 degrees', 0, 360, 0, key="2")
                st.write('Rotated', degrees, 'degrees')

                # download_button = st.download_button(
                #                 label="Download segmentated image",
                #                 data=segmentated,
                #                 file_name="test",
                #                 mime="image/png"
                #             )
              
              

    
if __name__ == "__main__":
    main()