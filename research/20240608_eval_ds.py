# %%
import start
import h5py
import matplotlib.pyplot as plt
import numpy as np

# %%
data_path = "data/P01"
train_files = h5py.File(f"{data_path}/train.hdf5", "r")
val_files = h5py.File(f"{data_path}/test.hdf5", "r")
test_files = h5py.File(f"{data_path}/post_pet_diag.hdf5", "r")

# %%
# For train

train_images = train_files["X_nii"]
train_labels = train_files["y"]
mean_train_image = np.mean(train_images, axis=0)
# %%
# For val
val_images = val_files["X_nii"]
val_labels = val_files["y"]
mean_val_image = np.mean(val_images, axis=0)

# %%
# For test
test_images = test_files["X_nii"]
test_labels = test_files["y"]
mean_test_image = np.mean(test_images, axis=0)


# %%
def show_histogram(image_np):
    plt.hist(image_np[:, 50, :].flatten(), bins=50)
    plt.title("Histogram of pixel values")
    plt.show()


# %%
show_histogram(mean_train_image)

# %%
show_histogram(mean_val_image)
# %%
show_histogram(mean_test_image)
# %%
special_train_image = [
    train_files["X_nii"][i]
    for i in range(len(train_files["X_nii"]))
    if train_files["ID"][i] == b"I346774"
][0]

# %%
show_histogram(special_train_image)
# %%
other_special_train_image = [
    train_files["X_nii"][i]
    for i in range(len(train_files["X_nii"]))
    if train_files["ID"][i] == b"I72373"
][0]
