# %%
from research import start  # noqa
import h5py
import matplotlib.pyplot as plt
import numpy as np

# %%
data_path = "data/preproc"
train_file = h5py.File(f"{data_path}/train_csv.hdf5", "r")
test_file = h5py.File(f"{data_path}/test_csv.hdf5", "r")

# %%
# check if there is any NaN in train files
train_images = train_file["X_nii"]
train_labels = train_file["y"]
print(np.isnan(train_images).sum())
# %%
# Replace NaN with 0
train_images = np.nan_to_num(train_images)
