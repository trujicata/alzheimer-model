#%%
# import start
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

#%%
def show_atlas(img_np):
    for i in range(img_np.shape[2]):
        plt.imshow(img_np[:, :, i], cmap="gray")
        plt.colorbar()
        plt.title("PET Image")
        plt.show()


# %%
atlas = nib.load(
    "../data/cerebra/CerebrA.nii"
)
atlas_np = np.array(atlas.dataobj)

#%%
# show_atlas(atlas_np)
#%%
labels = pd.read_csv('../data/cerebra/CerebrA_LabelDetails.csv')
# %%
# Display the image using Matplotlib
def show_nii(img_np):
    for i in range(img_np.shape[2]):
        plt.imshow(img_np[:, :, i, 0], cmap="gray")
        plt.colorbar()
        plt.title("PET Image")
        plt.show()

#%%
# show_nii(img_np)
#%%
print(np.shape(img_np))
print(np.shape(atlas_np))
# %%
imageh5 = h5py.File('../data/test.hdf5','r')
# %%
img_1 = imageh5["X"][10,:,:,:]
# from matplotlib import pyplot as plt
# plt.imshow(a_slice, cmap="gray")
# %%
show_atlas(img_1[:,:,50:55])
# %%
def find_slices_with_value(arr, x):
    slices_with_value = []
    for slice_idx in range(arr.shape[2]):
        slice_data = arr[:, :, slice_idx]
        if np.any(slice_data == x):
            slices_with_value.append(slice_idx)
    return slices_with_value
# %%
label_x = 50
print(find_slices_with_value(atlas_np,label_x))
slices_label_x = find_slices_with_value(atlas_np, label_x)
#%%
icbm152 = nib.load(
    "../data/icbm_avg_152_t1_tal_lin.nii"
)
icbm152_np = np.array(icbm152.dataobj)
# %%
# Crop or pad the image to match the shape of the segmented mask
if atlas_np.shape != icbm152_np.shape:
    min_shape = np.min([atlas_np.shape, icbm152_np.shape], axis=0)
    atlas_np = atlas_np[:min_shape[0], :min_shape[1], :min_shape[2]]
    icbm152_np = icbm152_np[:min_shape[0], :min_shape[1], :min_shape[2]]

masked_image = np.where(atlas_np == label_x, icbm152_np, 0)
show_atlas(masked_image[:,:,slices_label_x])
# %%
