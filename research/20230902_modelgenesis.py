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
# %%
base_model
# %%
base_model.down_tr128
# %%
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x):
        out = self.base_model.down_tr64(x) #torch.Size([1, 64, 54, 45, 45])
        out = self.base_model.down_tr128(out[0]) #torch.Size([1, 128, 27, 22, 22])
        out = self.base_model.down_tr256(out[0]) #torch.Size([1, 256, 13, 11, 11])
        out = self.base_model.down_tr512(out[0]) #torch.Size([1, 512, 13, 11, 11])
        return out

encoder = Encoder(base_model)
del base_model
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
embeddings = []
for i, sample in enumerate(train_data[:10]):
    print("Embedding # {}".format(i))
    tensor_sample = to_tensor(sample)
    embedding = encoder(tensor_sample.unsqueeze(0).unsqueeze(0))
    embeddings.append(embedding[0])
#%%
labels = []
for y in train_labels[:10]:
    labels.append(y)
# %%
import os
import json
os.makedirs("data/embeddings", exist_ok=True)
torch.save(embeddings, "data/embeddings/Genesis_Chest_CT.pt")
#%%
torch_embeddings = torch.stack(embeddings, dim=0).squeeze(1)
print(torch_embeddings.shape)
#%%
from torch.utils.tensorboard import SummaryWriter

labs = labels

writer = SummaryWriter(log_dir=".runs/embeddings")

writer.add_embedding(torch_embeddings, metadata=labs)