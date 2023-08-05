# %%
import start  # noqa
import numpy as np
from models.classifier3D.model import Classifier3D
from models.core.dataset import ADNIDataModule

# %%
datamodule = ADNIDataModule(
    train_path="/home/brainers-adni/train.hdf5", val_path="/home/brainers-adni/test.hdf5", num_workers=1
)

# %%
datamodule.setup("eval")

# %%
train_dataset = datamodule.train_dataset
# %%
sample = train_dataset[0]
img = sample["image"]

# Save img as a npy file
np.save("sample.npy", img.numpy())
# %%
import torch.nn as nn

# %%
module = nn.Sequential(
    nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=0),
    nn.LeakyReLU(),
    nn.MaxPool3d((2, 2, 2)),
)

# %%
module(sample["image"].unsqueeze(0)).shape
# %%
model = Classifier3D()
model.eval()
# %%
model(sample["image"].unsqueeze(0).unsqueeze(0)).shape
# %%
import time

start = time.time()
hola = np.load("sample.npy")
print(time.time() - start)
# %%
