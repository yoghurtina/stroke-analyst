import cv2
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
from module.segmentation import segmentation, get_segmented
from module.alignment import alignment, is_aligned
from PIL import Image
import io
from module.normalization import equalize_this

st.header("Rotation Correction of the section")

def rotate_image(image, degrees):
    """
    Rotates the image by the specified number of degrees
    """
    rotated_image = image.rotate(degrees)
    return rotated_image

image_from_previous_step = Image.open("src/temp/segmentation/segmented_image.jpg")
mask_from_previous_step =  Image.open("src/temp/segmentation/mask_bgs.jpg") 

st.image([image_from_previous_step, mask_from_previous_step], width=200, caption=["Previously Background Segmented Section", "Mask of previously Segmented Section"])

def main():
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    for file in uploaded_files:
        if file:
            file_bytes = io.BytesIO(file.read())
            image = Image.open(file_bytes)

            st.image(image_from_previous_step, width=300)
            segmented_image = get_segmented("src/temp/segmentation/source_image.jpg", "src/temp/segmentation/mask_bgs.jpg")
            segmented_pil_image  =Image.fromarray(segmented_image)
            segmented_pil_image.save("src/temp/segmentation/segmented_image.jpg")
            aligned = alignment(segmented_image, np.array(mask_from_previous_step))

            aligned = Image.fromarray(aligned)
            st.image(aligned, caption="Aligned Section",width=300)

            rotation_degrees = st.number_input('Degrees to rotate the image (if needed):', min_value=-360, max_value=360, value=0, step=1)
            st.write('The current rotation degree is ', rotation_degrees)
            rotated_image = rotate_image(aligned, rotation_degrees)

            st.image(rotated_image, caption="Rotated Image", width=300)

            rotated_image_array = np.array(rotated_image)
            rotated_pil_image = Image.fromarray(rotated_image_array)
            rotated_pil_image.save("src/temp/alignment/aligned_section.jpg")
            equalized_array = equalize_this("src/temp/alignment/aligned_section.jpg")
            print(type(equalized_array))
            equalized_array = equalized_array.astype('uint8')
            equalized_pil_image = Image.fromarray(equalized_array)
            equalized_pil_image.save("src/temp/alignment/equalized_aligned_section.jpg")
        
if __name__ == "__main__":
    main()


