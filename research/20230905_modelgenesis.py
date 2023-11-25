#%%
import start
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ModelsGenesis.pytorch.unet3d as unet3d

# %%
base_model = unet3d.UNet3D()
# %%
weight_dir = 'models/weights/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir, map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
base_model.load_state_dict(unParalled_state_dict)
base_model

# %%
from models.unet3D.unet_encoder import UNet3DEncoder

encoder = UNet3DEncoder()

# %%
encoder_state_dict={}
for k,v in unParalled_state_dict.items():
    if k.startswith('down_tr'):
        encoder_state_dict[k] = v
encoder.load_state_dict(encoder_state_dict)
# %%
encoder
# %%
import h5py

train_path = "data/train.hdf5"
test_path = "data/test.hdf5"
holdout_path = "data/holdout.hdf5"
# %%
train_h5 = h5py.File(train_path, "r")
# test_h5 = h5py.File(test_path, "r")
# holdout_h5 = h5py.File(holdout_path, "r")
# %%
train_data = train_h5["X_nii"]
train_labels = train_h5["y"]
# %%
sample = train_data[15]
sample.shape
# %%
from torchvision import transforms as T
to_tensor = T.ToTensor()
# %%
tensor_sample = to_tensor(sample)
tensor_sample.shape
#%%
out = encoder(tensor_sample.unsqueeze(0).unsqueeze(0))
# %%
# Save weights
torch.save(encoder.state_dict(), 'models/weights/Genesis_Chest_CT_encoder.pt')
# %%
out.shape
# %%
