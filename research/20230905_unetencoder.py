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
train_path = "data/train.hdf5"
train_h5 = h5py.File(train_path, "r")
train_data = train_h5["X_nii"]
sample = train_data[15]
sample.shape
#%%
to_tensor = T.ToTensor()
tensor_sample = to_tensor(sample)
tensor_sample.shape
#%%
out = threed_encoder(tensor_sample.unsqueeze(0).unsqueeze(0))
out.shape
# %%
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(
            512, 1024, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(
            1024, 2048, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)
        )
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(
            2048, 2048, kernel_size=(4, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(
            2048, 1024, kernel_size=4, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose3d(
            1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose3d(
            512, 512, kernel_size=(6, 4, 4), stride=2, padding=2, output_padding=1
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create an instance of the autoencoder
autoencoder = Autoencoder()

sample = torch.rand(1, 512, 13, 11, 11)
out1 = autoencoder.encoder(sample)
print("Encoder output shape:")
print(out1.shape)
out2 = autoencoder(sample)
print("Decoder output shape:")
print(out2.shape)

# %%
from models.unet3D.model import Autoencoder

autoencoder = Autoencoder()
# %%
label = out
pred = autoencoder(out)
#%%
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError

r2 = R2Score()
mae = MeanAbsoluteError()
mse = MeanSquaredError()

# %%
r2(pred, label)
# %%
mae(pred, label)
# %%
mse(pred, label)

# %%
from torchmetrics import StructuralSimilarityIndexMeasure

ssim = StructuralSimilarityIndexMeasure()
# %%
ssim(pred, label)
# %%
