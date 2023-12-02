#%%
import start
import h5py

import torch
from pmrc.SwinUNETR.BTCV.swinunetr import SwinUnetrModelForInference
# %%
model = SwinUnetrModelForInference.from_pretrained('darragh/swinunetr-btcv-tiny')
# %%
train_path = "data/train.hdf5"
test_path = "data/test.hdf5"
holdout_path = "data/holdout.hdf5"
# %%
train_h5 = h5py.File(train_path, "r")
test_h5 = h5py.File(test_path, "r")
holdout_h5 = h5py.File(holdout_path, "r")
# %%
train_data = train_h5["X_nii"]
# %%
sample = train_data[15]
sample.shape
# %%
from torchvision import transforms as T
to_tensor = T.ToTensor()
# %%
tensor_sample = to_tensor(sample)
tensor_sample.shape
# %%
model(tensor_sample.unsqueeze(0))
# %%
