# %%
from research import start  # noqa
import h5py
import matplotlib.pyplot as plt
import numpy as np

# %%
classes = ["AD", "MCI", "CN"]
data_path = "data/P01"
val_file = h5py.File(f"{data_path}/test_csv.hdf5", "r")
test_file = h5py.File(f"{data_path}/pre_pet_diag.hdf5", "r")
train_file = h5py.File(f"{data_path}/train_csv.hdf5", "r")
# %%
# Check the labels of the training set
train_labels = train_file["y"]
label_count_train = {0: 0, 1: 0, 2: 0}
for label in train_labels:
    label_count_train[label] += 1
print(label_count_train)

# %%
# Check the labels of the validation set
val_labels = val_file["y"]
label_count_val = {0: 0, 1: 0, 2: 0}
for label in val_labels:
    label_count_val[label] += 1
print(label_count_val)
# %%
# Check the labels of the test set
test_labels = test_file["y"]
label_count_test = {0: 0, 1: 0, 2: 0}
for label in test_labels:
    label_count_test[label] += 1
label_count_test

# %%
