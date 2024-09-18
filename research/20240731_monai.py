# %%
from research import start  # noqa
import torch
import monai

# %%
checkpoint = torch.load("data/model3.pth")

# %%
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=3)
model.load_state_dict(checkpoint)

# %%
model
# %%
