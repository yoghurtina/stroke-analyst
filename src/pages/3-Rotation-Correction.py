import cv2
import numpy as np
import streamlit as st
from streamlit_image_comparison import image_comparison
from module.utils import segmentation_old, get_segmented
from module.alignment import alignment, is_aligned
from PIL import Image
import io, os
from module.utils import equalize_this, create_superpixel_image

st.header("Rotation Correction of the section")

def rotate_image(image, degrees):
    """
    Rotates the image by the specified number of degrees
    """
    rotated_image = image.rotate(degrees)
    return rotated_image

image_from_previous_step = Image.open("results/segmentation/segmented_image.jpg")
mask_from_previous_step =  Image.open("results/segmentation/mask2_bgs.jpg") 

st.image([image_from_previous_step, mask_from_previous_step], width=200, caption=["Previously Background Segmented Section", "Mask of previously Segmented Section"])


def main():
    # Check if the file exists before trying to open it
    segmented_image_path = "results/segmentation/segmented_image.jpg"
    mask_path = "results/segmentation/mask_bgs.jpg"
    if os.path.exists(segmented_image_path) and os.path.exists(mask_path):
        segmented_image = Image.open(segmented_image_path)
        mask = Image.open(mask_path)

        segmented_image = get_segmented("results/segmentation/source_image.jpg", "results/segmentation/mask_bgs.jpg")
        segmented_pil_image  =Image.fromarray(segmented_image)
        segmented_pil_image.save("results/segmentation/segmented_image.jpg")

        aligned = alignment(segmented_image, np.array(mask_from_previous_step))
        aligned = Image.fromarray(aligned)
        st.image(aligned, caption="Rotated Section",width=200)

        st.write('**In case the section is not accurately aligned to the x axis, please correct the alignment.**')

        rotation_degrees = st.number_input('Degrees to rotate the section (-360 to 360):', min_value=-360, max_value=360, value=0, step=1)
        st.write('The current rotation degree is ', rotation_degrees)
        rotated_image = rotate_image(aligned, rotation_degrees)

        st.image(rotated_image, caption="Rotated Image", width=200)

        rotated_image_array = np.array(rotated_image)
        rotated_pil_image = Image.fromarray(rotated_image_array)
        rotated_pil_image.save("results/alignment/aligned_section.jpg")
        equalized_array = equalize_this("results/alignment/aligned_section.jpg")
        print(type(equalized_array))
        equalized_array = equalized_array.astype('uint8')
        equalized_pil_image = Image.fromarray(equalized_array)
        equalized_pil_image.save("results/alignment/equalized_aligned_section.jpg")
        # sp = create_superpixel_image("src/temp/alignment/equalized_aligned_section.jpg")
        # st.image(sp)
        st.success(f"Correct aligned section successfully saved!")

    else:
        st.error("Required images do not exist. Please upload them in the previous steps.")     

if __name__ == "__main__":
    main()


