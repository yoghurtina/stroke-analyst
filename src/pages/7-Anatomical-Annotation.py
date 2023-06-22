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
from module.region_naming import region_naming
from module.compute_voldata import compute_volumetric_data
import json

st.header("Annatomical Region Annotation and Volumetric Data")

# Load JSON data from file and convert to dictionary
with open('/home/ioanna/Documents/Thesis/raw_data/acronyms.json') as file:
    json_data = json.load(file)
    # json_data = {item['id']: item for item in json_data}
# print(json_data)

lesion_AS = Image.open('/home/ioanna/Documents/Thesis/src/temp/detection/mask3_hem1.jpg')
affected_regions = region_naming(json_data, -2.68e-3, lesion_AS, '/home/ioanna/Documents/Thesis/src/temp/anatomy')

print(type(affected_regions))

# list to dataframe
df = pd.DataFrame(affected_regions, columns=['Affected Regions'])
st.write(df)

vol_data = compute_volumetric_data("/home/ioanna/Documents/Thesis/src/temp/segmentation/mask_segmented_image.jpg","/home/ioanna/Documents/Thesis/src/temp/detection/mask3_hem1.jpg",\
                         "/home/ioanna/Documents/Thesis/src/temp/detection/mask1_hem1.jpg", "/home/ioanna/Documents/Thesis/src/temp/detection/mask1_hem2.jpg", "/home/ioanna/Documents/Thesis/src/temp")

st.write("Volumetric data computed!")

vol_data = pd.read_csv("/home/ioanna/Documents/Thesis/src/temp/results.csv")
st.write(vol_data)

df2 =pd.DataFrame(vol_data)
df2 = df2.transpose()
df2.columns = df2.iloc[0]
df2 = df2.drop(df2.index[0])
# df2 = df2.reset_index()
# df2 = df2.rename(columns={"index": "Affected Regions"})
st.write(df2)
