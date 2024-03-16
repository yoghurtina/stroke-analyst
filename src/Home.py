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

TODO: 
    -superpixels
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

The analysis process encompasses the following tasks:

1. **Mapping**: Establishing the spatial correspondence between a coronary section and its corresponding section in the Allen Atlas.
2. **Background Segmentation**: Separating the background elements from the coronary section to obtain a clean representation.
3. **Rotation Correction**: Rectifying the rotation of the coronary section, if required, to align it properly.
4. **Registration**: Performing registration between the uploaded coronary section and the corresponding atlas section, assessing the quality of alignment.
5. **Stroke Lesion Detection**: Utilizing advanced algorithms, identifying the stroke lesion within the coronary section and visualizing the extracted masks.
6. **Anatomical Annotation**: Determining the affected anatomical regions by associating the identified stroke lesion with the relevant anatomical structures 
using the generated lesion mask. Furthermore, volumetric data of the stroke can be computed and analyzed.
"""

