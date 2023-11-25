# %%
import start

import h5py
import torch
from captum.attr import IntegratedGradients
from torchvision import transforms as T

from models.classifier3D.model import Classifier3D

# %%

model = Classifier3D()
# %%
weights = torch.load(
    "lightning_logs/checkpoints/convnet3d-from-checkpoint/convnet3d-from-checkpoint-epoch=66-val_loss=0.35-val_f1=0.94.ckpt"
)["state_dict"]
weights

# %%
model.load_state_dict(weights)
# %%
torch.manual_seed(123)
train_path = "data/new-norm/train.hdf5"
train_h5 = h5py.File(train_path, "r")
train_data = train_h5["X_nii"]
input = train_data[15]
input.shape

# %%
to_tensor = T.ToTensor()
input = to_tensor(input).unsqueeze(0).unsqueeze(0)
baseline = torch.zeros_like(input)
input.shape

# %%
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(
    input, baseline, target=0, return_convergence_delta=True
)
print("IG Attributions:", attributions)
print("Convergence Delta:", delta)

# %%
from matplotlib import pyplot as plt

index = 25
plt.imshow(attributions[0, 0, index, :, :].detach().numpy(), cmap="bwr")
# Add scale bar
plt.colorbar()
plt.show()
plt.imshow(input[0, 0, index, :, :].detach().numpy(), alpha=0.5, cmap="gray")
plt.show()

# %%
