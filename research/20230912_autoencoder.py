#%%
import start
import torch

from models.unet_autoencoder.model import Autoencoder, Encoder

# %%
model = Autoencoder()
checkpoint = torch.load(
    "lightning_logs/autoencoder/autoencoder/autoencoder_21:55_checkpoints/autoencoder_21:55-epoch=234-val_loss=0.03-val_mae=0.12-val_mse=0.03-val_ssim=0.87.ckpt"
)
model.load_state_dict(checkpoint["state_dict"])
# %%
encoder = Encoder()
encoder.load_state_dict(model.encoder.state_dict())
# %%
torch.save(encoder.state_dict(), "models/weights/encoder_autoencoder.pt")
# %%
