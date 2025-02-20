import pandas as pd
from PIL import Image
import streamlit as st
from module.region_naming import region_naming
from module.compute_voldata import compute_volumetric_data
import json

st.header("Anatomical region annotation and volumetric calculations")

# Load JSON data from file and convert to dictionary
with open('raw_data/acronyms.json') as file:
    json_data = json.load(file)
    # json_data = {item['id']: item for item in json_data}
# print(json_data)

lesion_AS = Image.open('results/detection/mask3_hem1.jpg')
affected_regions = region_naming(json_data, -2.68e-3, lesion_AS, 'results/anatomy')

print(type(affected_regions))

# list to dataframe
df = pd.DataFrame(affected_regions, columns=['Affected Regions'])
st.write(df)

vol_data = compute_volumetric_data("results/segmentation/mask_2_bgs.jpg","results/detection/mask1_hem1.jpg",\
                         "results/alignment/mask_left_hemisphere.jpg", "results/alignment/mask_right_hemisphere.jpg", "results")

st.write("Volumetric data computed!")

vol_data = pd.read_csv("results/results.csv")
# st.write(vol_data)

df2 =pd.DataFrame(vol_data)
df2 = df2.transpose()
df2.columns = df2.iloc[0]
df2 = df2.drop(df2.index[0])
# df2 = df2.reset_index()
# df2 = df2.rename(columns={"index": "Affected Regions"})
st.write(df2)