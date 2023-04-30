import numpy as np
import streamlit as st
from module.segmentation import segmentation
import base64
import time
import uuid
from io import BytesIO
from pathlib import Path
import re
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("Image Segmentation")

# def png_export():
#     st.markdown(
#         """
#     Realtime update is disabled for this demo. 
#     Press the 'Download' button at the bottom of canvas to update exported image.
#     """
#     )
#     try:
#         Path("tmp/").mkdir()
#     except FileExistsError:
#         pass

#     # Regular deletion of tmp files
#     # Hopefully callback makes this better
#     now = time.time()
#     N_HOURS_BEFORE_DELETION = 1
#     for f in Path("tmp/").glob("*.png"):
#         st.write(f, os.stat(f).st_mtime, now)
#         if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
#             Path.unlink(f)

#     # Initialization
#     if 'key' not in st.session_state:
#         st.session_state['key'] = 'value'

#     if st.session_state["key"] == "":
#         st.session_state["key"] = re.sub(
#             "\d+", "", str(uuid.uuid4()).replace("-", "")
#         )

#     button_id = st.session_state["key"]
#     file_path = f"tmp/{button_id}.png"

#     custom_css = f""" 
#         <style>
#             #{button_id} {{
#                 display: inline-flex;
#                 align-items: center;
#                 justify-content: center;
#                 background-color: rgb(255, 255, 255);
#                 color: rgb(38, 39, 48);
#                 padding: .25rem .75rem;
#                 position: relative;
#                 text-decoration: none;
#                 border-radius: 4px;
#                 border-width: 1px;
#                 border-style: solid;
#                 border-color: rgb(230, 234, 241);
#                 border-image: initial;
#             }} 
#             #{button_id}:hover {{
#                 border-color: rgb(246, 51, 102);
#                 color: rgb(246, 51, 102);
#             }}
#             #{button_id}:active {{
#                 box-shadow: none;
#                 background-color: rgb(246, 51, 102);
#                 color: white;
#                 }}
#         </style> """

#     data = st_canvas(update_streamlit=False, key="png_export")
#     if data is not None and data.image_data is not None:
#         img_data = data.image_data
#         im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
#         im.save(file_path, "PNG")

#         buffered = BytesIO()
#         im.save(buffered, format="PNG")
#         img_data = buffered.getvalue()
#         try:
#             # some strings <-> bytes conversions necessary here
#             b64 = base64.b64encode(img_data.encode()).decode()
#         except AttributeError:
#             b64 = base64.b64encode(img_data).decode()

#         dl_link = (
#             custom_css
#             + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
#         )
#         st.markdown(dl_link, unsafe_allow_html=True)


def main():
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    for file in uploaded_files:
        if file:
            image = st.image(file,width=300)
                # Initialization
            if 'button' not in st.session_state:
                st.session_state['button'] = 'value'
            if st.button('Segmentate image', key = "value"):
            # for file in uploaded_files:
                seg, mask = segmentation(file)
                # frame = np.array(image)
                segmentated = st.image(seg,width=600)
                # mask = st.image(mask,width=600)
                # png_export()

    
if __name__ == "__main__":
    main()