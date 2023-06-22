import pandas as pd
from PIL import Image
import numpy as np

def compute_volumetric_data(whole_mask, lesion_mask, left_hem_mask, right_hem_mask, save_dir):
    pix_area = 0.021 * 0.021
    whole_mask = Image.open(whole_mask)
    lesion_mask = Image.open(lesion_mask)
    left_hem_mask = Image.open(left_hem_mask)
    right_hem_mask = Image.open(right_hem_mask)
    
    mask_array = np.array(whole_mask)
    lesion_array = np.array(lesion_mask)
    left_hem_array = np.array(left_hem_mask)
    right_hem_array = np.array(right_hem_mask)

    y = [i for i, val in enumerate(mask_array.flatten()) if val != 0]
    mask_area = len(y) * pix_area

    y = [i for i, val in enumerate(lesion_array.flatten()) if val != 0]
    lesion_area = len(y) * pix_area

    y = [i for i, val in enumerate(left_hem_array.flatten()) if val != 0]
    lh_area = len(y) * pix_area

    y = [i for i, val in enumerate(right_hem_array.flatten()) if val != 0]
    rh_area = len(y) * pix_area

    variable = ['whole_mask_area','lesion_area', 'left_hem_area', 'right_hem_area']
    value = [ mask_area, lesion_area, lh_area, rh_area]

    data = {'Variable': variable, 'Value': value}
    df = pd.DataFrame(data)
    print(df)

    df.to_csv(save_dir + '/results.csv', index=False)


def compute_volumetric_data(whole_mask, lesion_mask, left_hem_mask, right_hem_mask, save_dir):
    pix_area = 0.021 * 0.021
    whole_mask = Image.open(whole_mask)
    lesion_mask = Image.open(lesion_mask)
    left_hem_mask = Image.open(left_hem_mask)
    right_hem_mask = Image.open(right_hem_mask)

    mask_array = np.array(whole_mask)
    lesion_array = np.array(lesion_mask)
    left_hem_array = np.array(left_hem_mask)
    right_hem_array = np.array(right_hem_mask)

    y = [i for i, val in enumerate(mask_array.flatten()) if val != 0]
    mask_area = len(y) * pix_area

    y = [i for i, val in enumerate(lesion_array.flatten()) if val != 0]
    lesion_area = len(y) * pix_area

    y = [i for i, val in enumerate(left_hem_array.flatten()) if val != 0]
    lh_area = len(y) * pix_area

    y = [i for i, val in enumerate(right_hem_array.flatten()) if val != 0]
    rh_area = len(y) * pix_area

    # Calculate volumes
    slice_thickness = 0.021  # Adjust according to the actual thickness of the slice
    lh_volume = lh_area * slice_thickness
    rh_volume = rh_area * slice_thickness
    infarct_volume = lesion_area * slice_thickness

    # Calculate percentage volumes
    total_volume = lh_volume + rh_volume + infarct_volume
    v_inf = ((rh_volume + infarct_volume - lh_volume) / rh_volume) * 100
    v_he = ((lh_volume - rh_volume) / rh_volume) * 100

    # Create dataframe
    variable = ['whole_mask_area', 'lesion_area', 'left_hem_area', 'right_hem_area', 'left_hem_volume', 'right_hem_volume', 'infarct_volume', 'v_inf', 'v_he']
    value = [mask_area, lesion_area, lh_area, rh_area, lh_volume, rh_volume, infarct_volume, v_inf, v_he]
    data = {'Variable': variable, 'Value': value}
    df = pd.DataFrame(data)

    # Print and save the results
    print(df)
    df.to_csv(save_dir + '/results.csv', index=False)
