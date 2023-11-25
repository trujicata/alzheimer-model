#%%
import start
import torch
import h5py
from torchvision import transforms as T
from models.unet_autoencoder.unet_encoder import UNet3DEncoder

#%%
threed_encoder = UNet3DEncoder()

weights = torch.load(
    "models/weights/Genesis_Chest_CT_encoder.pt", map_location=torch.device("cpu")
)
threed_encoder.load_state_dict(weights)
# %%
tensor_sample = torch.randn(90, 90, 86)
#%%
out = threed_encoder(tensor_sample.unsqueeze(0).unsqueeze(0))
out.shape
# %%
