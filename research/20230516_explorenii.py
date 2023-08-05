#%%
import start
import nibabel as nib
import numpy as np
from PIL import Image

#%%
# Single image to numpy
img_1 = nib.load(
    "data/PET/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165151377_159_S165970_I330092.nii"
)
#%%
img_np = np.array(img_1.dataobj)
# %%
import matplotlib.pyplot as plt

# Display the image using Matplotlib
def show_nii(img_np):
    for i in range(img_np.shape[2]):
        plt.imshow(img_np[:, :, i, 0], cmap="gray")
        plt.colorbar()
        plt.title("PET Image")
        plt.show()


#%%
show_nii(img_np)
# %%
# 1 . Sacar las capas de los extremos porque pueden hacer ruido
def peel_nii(img, num_layers):
    peeled_image = img[:, :, num_layers:-num_layers, :]
    return peeled_image


# %%
peeled = peel_nii(img_np, 3)
# %%
show_nii(peeled)
# %%
# 2. Bajar la profundidad
def scale_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = np.interp(arr, (min_val, max_val), (0, 255))
    scaled_arr = np.round(scaled_arr).astype(int)
    return scaled_arr


#%%
def scale_nii(img_np):
    for i in range(img_np.shape[2]):
        img_np[:, :, i, 0] = scale_array(img_np[:, :, i, 0])
    return img_np


# %%
scaled_peeled = scale_nii(peeled)

# %%
show_nii(scaled_peeled)
# %%
# 3. Hacer un promedio con todos los frames (siempre 3 primeros)
img_2 = nib.load(
    "/home/catalina/workspace/alzheimer-model/data/PET/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165149569_77_S165970_I330092.nii"
)
img2_np = np.array(img_2.dataobj)
img_3 = nib.load(
    "/home/catalina/workspace/alzheimer-model/data/PET/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165152374_36_S165970_I330092.nii"
)
img3_np = np.array(img_3.dataobj)

# %%
img_promedio = np.mean(np.array([img_np, img2_np, img3_np]), axis=0)
img_promedio.shape

# %%
show_nii(img_promedio)
# %%
peeled_promedio = peel_nii(img_promedio, 3)
show_nii(peeled_promedio)
# %%
scaled_peeled_promedio = scale_nii(peeled_promedio)
show_nii(scaled_peeled_promedio)
# %%
