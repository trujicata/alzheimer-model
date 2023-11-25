#%%
import start

import h5py
#%%
train_h5 = h5py.File("data/test_the_model/train.hdf5", "r")
val_h5 = h5py.File("data/test_the_model/test.hdf5", "r")

# %%
X_train = train_h5["X_nii"]
y_train = train_h5["y"]
X_val = val_h5["X_nii"]
y_val = val_h5["y"]
# %%
# Check amount of classes in train
import numpy as np
np.unique(y_train, return_counts=True)
# %%
# Check amount of classes in val
np.unique(y_val, return_counts=True)
# %%
# Eliminate class 1 from train
new_X_train = []
nex_y_train = []
for x in range(len(y_train)):
    if y_train[x] != 1:
        new_X_train.append(X_train[x])
        nex_y_train.append(y_train[x])
new_X_train = np.array(new_X_train)
new_y_train = np.array(nex_y_train)
# Transform into h5 file
with h5py.File("data/data-2-classes/train.hdf5", "w") as f:
    f.create_dataset("X_nii", data=new_X_train)
    f.create_dataset("y", data=nex_y_train)


# %%
# Eliminate class 1 from val
new_X_val = []
new_y_val = []

for x in range(len(y_val)):
    if y_val[x] != 1:
        new_X_val.append(X_val[x])
        new_y_val.append(y_val[x])
new_X_val = np.array(new_X_val)
new_y_val = np.array(new_y_val)

# Transform into h5 file
with h5py.File("data/data-2-classes/test.hdf5", "w") as f:
    f.create_dataset("X_nii", data=new_X_val)
    f.create_dataset("y", data=new_y_val)


# %%
# New dataset with less of class 1
new_X_train = []
nex_y_train = []
for x in range(len(y_train)):
    if y_train[x] != 1:
        new_X_train.append(X_train[x])
        nex_y_train.append(y_train[x])
    else:
        if np.random.rand() < 0.4:
            new_X_train.append(X_train[x])
            nex_y_train.append(y_train[x])

new_X_train = np.array(new_X_train)
new_y_train = np.array(nex_y_train)
# Transform into h5 file
with h5py.File("data/data-balance/train.hdf5", "w") as f:
    f.create_dataset("X_nii", data=new_X_train)
    f.create_dataset("y", data=nex_y_train)

# %%
new_X_val = []
new_y_val = []
for x in range(len(y_val)):
    if y_val[x] != 1:
        new_X_val.append(X_val[x])
        new_y_val.append(y_val[x])
    else:
        if np.random.rand() < 0.4:
            new_X_val.append(X_val[x])
            new_y_val.append(y_val[x])

new_X_val = np.array(new_X_val)
new_y_val = np.array(new_y_val)

# Transform into h5 file
with h5py.File("data/data-balance/test.hdf5", "w") as f:
    f.create_dataset("X_nii", data=new_X_val)
    f.create_dataset("y", data=new_y_val)
# %%
new_train_h5 = h5py.File("data/data-balance/train.hdf5", "r")
new_val_h5 = h5py.File("data/data-balance/test.hdf5", "r")


# %%
X_train = new_train_h5["X_nii"]
y_train = new_train_h5["y"]
X_val = new_val_h5["X_nii"]
y_val = new_val_h5["y"]
# %%
# Check amount of classes in train
import numpy as np
np.unique(y_train, return_counts=True)
# %%
# Check amount of classes in val
np.unique(y_val, return_counts=True)
# %%
# Get 5 random samples from each class
import random

x_samples = []
y_samples = []

for x in range(1, 5):
    random_index = random.randint(0, len(y_train))
    x_samples.append(X_train[random_index])
    y_samples.append(y_train[random_index])
# %%
x_samples = np.array(x_samples)
y_samples = np.array(y_samples)
# Transform into h5 file
with h5py.File("data/test_the_model/test.hdf5", "w") as f:
    f.create_dataset("X_nii", data=x_samples)
    f.create_dataset("y", data=y_samples)
# %%
