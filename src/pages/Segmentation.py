import cv2
import numpy as np
import streamlit as st
import image_segmentation as seg

def main():                         
    st.title("Image Segmentation App")

    # Upload file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if __name__ == "__main__":
    main()