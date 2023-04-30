import streamlit as st
from PIL import Image
import io
from streamlit_image_select import image_select
import numpy as np


"""
# Mapping with Atlas

"""
import os

def rotate_image(image, degrees):
    """
    Rotates the image by the specified number of degrees
    """
    file_bytes = io.BytesIO(file.read())
    image = Image.open(file_bytes)

    rotated_image = image.rotate(degrees)
    return rotated_image


def read_photos_from_folder(folder_path):
    """Reads all photos from a local folder."""
    photos = []
    contents = os.listdir(folder_path)
    # Sort the contents alphabetically
    contents.sort()
    for file_name in contents:
        if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
            photo_path = os.path.join(folder_path, file_name)
            photos.append(photo_path)
            # st.write(file_name)
    return photos


col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Upload a section", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    for file in uploaded_files:
        if file:
            image = st.image(file,width=300)

            rotation_degrees = st.number_input('Degrees to rotate the image:', min_value=-360, max_value=360, value=0, step=1)
            st.write('The current number is ', rotation_degrees)
            rotated_image = rotate_image(image, rotation_degrees)

            # Show rotated image
            st.image(rotated_image, caption="Rotated Image")
               

# with col2:
#     folder_path = '/home/ioanna/Documents/Thesis/raw_data/atlas'

#     # Read photos from the local folder
#     photos = read_photos_from_folder(folder_path)

#     # Display the photos in a carousel using Streamlit
#     if len(photos) > 0:
#         # carousel_index = st.slider("Choose corresponding atlas section", 0, len(photos)-1, 1)
#         # image = st.image(photos[carousel_index], use_column_width=True)
        
#         # if "rotate_slider" not in st.session_state:
#         #     st.session_state["rotate_slider"] = 0
#         # rotation_degrees = st.number_input('Degrees to rotate the image:', min_value=-360, max_value=360, value=st.session_state["rotate_slider"], key="rotate_slider")
#         img = image_select(
#             label="Select corresponding atlas section",
#             images=photos,
            
#         )
#         st.write(str(img))
#     else:
#         st.warning("No photos found in the specified folder.")


# with col2:
#     import streamlit as st
#     import streamlit.components.v1 as components

#     def main():

#         imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/frontend/public")

#         imageUrls = [
#             "https://images.unsplash.com/photo-1610016302534-6f67f1c968d8?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1075&q=80",
#             "https://images.unsplash.com/photo-1516550893923-42d28e5677af?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=872&q=80",
#             "https://images.unsplash.com/photo-1541343672885-9be56236302a?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
#             "https://images.unsplash.com/photo-1512470876302-972faa2aa9a4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
#             "https://images.unsplash.com/photo-1528728329032-2972f65dfb3f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
#             "https://images.unsplash.com/photo-1557744813-846c28d0d0db?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1118&q=80",
#             "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
#             "https://images.unsplash.com/photo-1595867818082-083862f3d630?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
#             "https://images.unsplash.com/photo-1622214366189-72b19cc61597?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
#             "https://images.unsplash.com/photo-1558180077-09f158c76707?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
#             "https://images.unsplash.com/photo-1520106212299-d99c443e4568?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
#             "https://images.unsplash.com/photo-1534430480872-3498386e7856?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
#             "https://images.unsplash.com/photo-1571317084911-8899d61cc464?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
#             "https://images.unsplash.com/photo-1624704765325-fd4868c9702e?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
#         ]
#         selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

#         if selectedImageUrl is not None:
#             st.image(selectedImageUrl)

#     if __name__ == "__main__":
#         main()
