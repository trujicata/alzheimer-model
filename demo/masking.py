
import numpy as np
import pandas as pd
import nibabel as nib
from torchvision import transforms as T

def mask_image(image, atlas, area_of_interest):
    """ 
    Get masked image of area of interest in PET or MRI, based on given atlas.
    Args:
        - image (numpy array): numpy array of the image. Must be registered and be the same dimensions of atlas
        - atlas (numpy array): atlas where each pixel has a numeric code corresponding to the area assigned to that pixel
        - area_of_interest (int): numeric code for the area of interest to be masked
    Returns:
        - masked image as numpy array where values outside of area of interest are 0 and values inside the area remain untouched.
    """
    return np.where(atlas == area_of_interest, image, 0)

def measure_intensity(image, atlas, area_of_interest, absolute=False):
    """
    Measure the mean intensity for the brain area of interest
    Args:
        - image (numpy array): numpy array of the image. Must be registered and be the same dimensions of atlas
        - atlas (numpy array): atlas where each pixel has a numeric code corresponding to the area assigned to that pixel
        - area_of_interest (int): numeric code for the area of interest to be masked
        - absolute (bool): whether to take the absolute value of the intensity
    Returns:
        - mean of intensity for all the pixels in the area (float)
    """
    masked_image = mask_image(image, atlas, area_of_interest)
    if absolute:
        return np.mean(np.abs(masked_image))
    return np.mean(masked_image)

def measure_importance(image, atlas, area_of_interest, absolute=False):
    """
    Measure the mean intensity for the brain area of interest
    Args:
        - image (numpy array): numpy array of the image. Must be registered and be the same dimensions of atlas
        - atlas (numpy array): atlas where each pixel has a numeric code corresponding to the area assigned to that pixel
        - area_of_interest (int): numeric code for the area of interest to be masked
        - absolute (bool): whether to take the absolute value of the intensity
    Returns:
        - mean of intensity for all the pixels in the area (float)
    """
    masked_image = mask_image(image, atlas, area_of_interest)
    if absolute:
        return np.mean(masked_image), np.mean(np.abs(masked_image))
    return np.mean(masked_image), np.mean(masked_image)

def process_image(image, attributions, area_labels):
    """
    Get measurements for all brain areas for an image (intensity and attributions)
    Args:
        - image  (tensor): input image of PET-FDG
        - attributions (tensor): values from captum integrated gradients
        - area_labels (dataframe): dataframe containing the area numbers and the name of the areas
    Returns:
        - dataframe 
    """
    atlas = nib.load("cerebra_100_120_100.nii")
    atlas_np = np.array(atlas.dataobj)
    transposed_atlas = np.transpose(atlas_np, (0, 2, 1))
    to_tensor = T.ToTensor()
    transposed_atlas = to_tensor(transposed_atlas).unsqueeze(0).unsqueeze(0)

    results = []

    for _, row in area_labels.iterrows():
        right_area = row['RH Labels']
        left_area = row['LH Labels']

        right_intensity = measure_intensity(image, transposed_atlas, right_area)
        right_importance, right_absolute_importance = measure_importance(attributions, transposed_atlas, right_area, absolute=True)

        left_intensity = measure_intensity(image, transposed_atlas, left_area)
        left_importance, left_absolute_importance = measure_importance(attributions, transposed_atlas, left_area, absolute=True)

        results.append({
            'area': row['Label Name'],
            'RH_intensity': right_intensity,
            'RH_importance': right_importance,
            'RH_absolute_importance': right_absolute_importance,
            'LH_intensity': left_intensity,
            'LH_importance': left_importance,
            'LH_absolute_importance': left_absolute_importance
        })

    result_df = pd.DataFrame(results)

    # COLUMN WISE NORMALIZATION
    for col in ['RH_intensity', 'LH_intensity', 'RH_importance', 'LH_importance', 'RH_absolute_importance', 'LH_absolute_importance']:
        max_value = result_df[col].max()
        result_df[col] = result_df[col] / max_value if max_value != 0 else result_df[col]

    return result_df
