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

# """
# # Steps to follow:
# 1. Mapping Sections with Atlas (Mapping page)
# 2. Background Segmentation of Section (Segmentation page)
# 3. Section Alignment with Axis(Alignment page)
# 4. Stroke Detection, hemisphere masks and stroke mask creation (Detection page)
# 4. Anatomical Mapping to Atlas (Anatomy Page)
# """
"""
# Steps to follow:
1. Mapping a coronary section with it's corresponding Allen Atlas section
2. Segmentation of the background of the section
3. Correction of the rotation of the coronary section (if needed)
4. Registration results of the uploaded section and it's corresponding atlas section
5. Detection of the stroke lesion as well as of the ischemic and healthy hemispheres and visualization of the extracted masks.
6. Anatomical naming of the stroked regions gained from the lesion mask
"""