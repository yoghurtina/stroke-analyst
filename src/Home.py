import streamlit as st
import pandas as pd
import numpy as np
from streamlit import session_state
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping


"""
# Stroke Analyst - Wiki
__Under Construction__
"""

"""
# Steps to follow:
1. Mapping Sections with Atlas (Mapping page)
2. Background Segmentation of Section (Segmentation page)
3. Section Alignment with Axis(Alignment page)
4. Section Registration to Atlas (Registration page)
"""

page1_data = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if page1_data:
    session_state.page1_image = page1_data.read()

# Page 2: Access image from session state
page2_image = session_state.page1_image
if page2_image:
    st.image(page2_image, caption="Image from Page 1")