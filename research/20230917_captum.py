#%%
import start

import h5py
import torch
from captum.attr import IntegratedGradients
from torchvision import transforms as T

from models.unet_autoencoder.unet_encoder import UNet3DEncoder

#%%
model = UNet3DEncoder()

weights = torch.load(
    "models/weights/Genesis_Chest_CT_encoder.pt", map_location=torch.device("cpu")
)
model.load_state_dict(weights)

model.eval()
# %%
torch.manual_seed(123)
train_path = "data/train.hdf5"
train_h5 = h5py.File(train_path, "r")
train_data = train_h5["X_nii"]
input = train_data[15]
input.shape
#%%
to_tensor = T.ToTensor()
input = to_tensor(input).unsqueeze(0).unsqueeze(0)
baseline = torch.zeros_like(input)
input.shape
#%%
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)
# %%
