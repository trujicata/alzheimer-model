# %%
import start
import h5py

# %%
train_path = "data/original/train.hdf5"
test_path = "data/original/test.hdf5"
holdout_path = "data/original/holdout.hdf5"
# %%
train_h5 = h5py.File(train_path, "r")
test_h5 = h5py.File(test_path, "r")
holdout_h5 = h5py.File(holdout_path, "r")
# %%
train_data = train_h5["X_nii"]
# %%
test_h5["X_nii"]
# %%
holdout_h5["X_nii"]
# %%
import matplotlib.pyplot as plt


# Display the image using Matplotlib
def show_nii(img_np):
    for i in range(img_np.shape[2]):
        plt.imshow(img_np[:, i, :], cmap="gray")
        plt.colorbar()
        plt.title("PET Image")
        plt.show()


# %%
one = train_data[0]
one
# %%
show_nii(one)
# %%
