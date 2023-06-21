import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from matplotlib.patches import Polygon
from module.detection import seg_anything
from module.normalization import equalize_this
import os, shutil
from io import BytesIO


st.header("Annatomical Region Annotation and Volumetric Data")
