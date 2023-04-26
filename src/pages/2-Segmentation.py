import cv2
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
from module.segmentation import segmentation
from module.alignment import alignment, is_aligned

st.title("Image Segmentation")

def main():
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    for file in uploaded_files:
        if file:
            image = st.image(file,width=300)
            if st.button('Segmentate image', key="1"):
            # for file in uploaded_files:
                seg, mask = segmentation(file)
                # frame = np.array(image)
                segmentated = st.image(seg,width=600)
                mask = st.image(mask,width=600)
                
            # download_button = st.download_button(
            #                     label="Download segmentated image",
            #                     data=seg,
            #                     file_name="test",
            #                     mime="image/png"
            #                 )
              
              

    
if __name__ == "__main__":
    main()