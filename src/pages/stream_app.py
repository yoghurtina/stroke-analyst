import cv2
import numpy as np
import streamlit as st
import image_segmentation as seg

def main():
    st.title("Image Segmentation App")

    # Upload file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Perform segmentation on button click
    if st.button("Segment"):
        if uploaded_file is not None:
            # Perform segmentation
            segmented_image = seg.segmentation(uploaded_file)

            # Show segmented image
            st.image(segmented_image, channels="BGR", use_column_width=True)

            # Allow user to download the segmented image
            st.markdown("## Download Segmented Image")
            href = f'<a href="data:file/png;base64,{segmented_image.tobytes().hex()}" download="segmented_image.png">Download</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()