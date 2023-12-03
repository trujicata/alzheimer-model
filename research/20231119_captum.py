# %%
import start

import h5py
import torch
from captum.attr import IntegratedGradients
from torchvision import transforms as T

from models.classifier3D.model import Classifier3D

# %%

model = Classifier3D()
# %%
weights = torch.load(
    "models/weights/best-model-so-far.ckpt",
    map_location=torch.device("cpu"),
)["state_dict"]
weights

# %%
for k, v in weights.items():
    if k in model.state_dict().keys():
        print(k, " loaded")
        model.state_dict()[k] = v
    else:
        print(k, " not loaded")
# %%
torch.manual_seed(123)
train_path = (
    "/Volumes/Carlito/ADNI/Registered_normalizado_FINAL/train.hdf5"
)
train_h5 = h5py.File(train_path, "r")
train_data = train_h5["X_nii"]
input = train_data[1]
input.shape

# %%
to_tensor = T.ToTensor()
input = to_tensor(input).unsqueeze(0).unsqueeze(0)
baseline = torch.zeros_like(input)
input.shape

# %%
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(
    input, baseline, target=0, return_convergence_delta=True
)
print("IG Attributions:", attributions)
print("Convergence Delta:", delta)

# %%
from matplotlib import pyplot as plt

index = 75
plt.imshow(attributions[0, 0, index, :, :].detach().numpy(), cmap="bwr")
# Add scale bar
plt.colorbar()
plt.show()
plt.imshow(input[0, 0, index, :, :].detach().numpy(), alpha=0.5, cmap="gray")
plt.show()

#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib

def slices_with_area(atlas, area_of_interest):
    """
    Get the slices that have at least one pixel assigned to the area of interest
    Args:
        - atlas (numpy array): array where each pixel is assigned to a brain area
        - area_of_interest (int): numeric code for the area of interest
    Returns:
        - list with slices that have at least one pixel of the area of interest
    """
    slices_with_value = []
    for slice_idx in range(atlas.shape[4]):
        slice_data = atlas[0, 0, :, :, slice_idx]
        if area_of_interest in slice_data:
            slices_with_value.append(slice_idx)
    return slices_with_value

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

def measure_image(image, atlas, area_of_interest):
    """
    Measure the mean intensity and standard deviation for the brain area of interest
    Args:
        - image (numpy array): numpy array of the image. Must be registered and be the same dimensions of atlas
        - atlas (numpy array): atlas where each pixel has a numeric code corresponding to the area assigned to that pixel
        - area_of_interest (int): numeric code for the area of interest to be masked
    Returns:
        - mean of intensity for all the pixels in the area (int)
        - standard deviation of intensity for all the pixels in the area (int)
    """
    masked_image = mask_image(image, atlas, area_of_interest)
    slices_to_measure = slices_with_area(atlas, area_of_interest)
    mean_intensity = np.mean(np.abs(masked_image[0,0,:,:,slices_to_measure]))
    return mean_intensity


def process_h5(h5, atlas, areas_to_measure):
    """
    Get measurements (intensity and std) for all brain areas specified for all images in the h5 file
    Args:
        - h5 (h5 file): h5 file containing all registered images as numpy arrays
        - atlas
        - areas_to_measure (list): list of areas to measure according to numeric coding of atlas
    Returns:
        - dataframe containing all the images as rows, first column is the label, other columns represent the measurements (intensity and std) for different brain areas specified
    """
    result = pd.DataFrame({})
    avg_attributions = None
    avg_abs_attributions = None
    
    # Use tqdm for the loop to create a progress bar
    for i in tqdm(range(0, h5["X_nii"].shape[0]), desc="Processing"):
        row = {}
        row['label'] = h5["y"][i]
        row['sex'] = h5["X_Sex"][i]
        row['age'] = h5["X_Age"][i]
        input = h5["X_nii"][i]
        to_tensor = T.ToTensor()
        input = to_tensor(input).unsqueeze(0).unsqueeze(0)
        baseline = torch.zeros_like(input)
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(
            input, baseline, target=0, return_convergence_delta=True
        )
        
        if avg_attributions is None:
            avg_attributions = attributions.clone()
        else:
            avg_attributions += attributions
        
        if avg_abs_attributions is None:
            avg_abs_attributions = torch.abs(attributions.clone())
        else:
            avg_abs_attributions += torch.abs(attributions)
        
        for area in areas_to_measure:
            intensity = measure_image(attributions, atlas, area)
            row[f'{area}'] = intensity
        
        new_df = pd.DataFrame([row])
        result = pd.concat([result, new_df], ignore_index=True)
    
    # Calculate average attributions
    avg_attributions /= len(result)
    
    # Calculate average absolute attributions
    avg_abs_attributions /= len(result)
    
    return result, avg_attributions, avg_abs_attributions

def get_captum_h5(train_path, test_path):
    atlas = nib.load(
        "cerebra_100_120_100.nii"
    )
    atlas_np = np.array(atlas.dataobj)
    transposed_atlas = np.transpose(atlas_np, (0, 2, 1))
    to_tensor = T.ToTensor()
    transposed_atlas = to_tensor(transposed_atlas).unsqueeze(0).unsqueeze(0)
    all_areas = np.unique(atlas_np)
    train_file = h5py.File(train_path)
    test_file = h5py.File(test_path)
    print('Start explainability calculation for Train dataset')
    train_captum_data = process_h5(train_file, transposed_atlas, all_areas)
    print('Start explainability calculation for Test dataset')
    test_captum_data = process_h5(test_file, transposed_atlas, all_areas)
    train_captum_data.to_csv('train_captum_FDG.csv')
    test_captum_data.to_csv('test_captum_FDG.csv')
    print('Ended explainability calculation from train and test h5 files')
# %%
train_h5_path = '/Volumes/Carlito/ADNI/Registered_normalizado_FINAL/train.hdf5'
test_h5_path = '/Volumes/Carlito/ADNI/Registered_normalizado_FINAL/test.hdf5'
get_captum_h5(train_h5_path, test_h5_path) 

# %%
