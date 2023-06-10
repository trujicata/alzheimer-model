# %%
import start  # noqa
from models.classifier3D.model import Classifier3D
from models.core.dataset import ADNIDataModule

# %%
datamodule = ADNIDataModule(
    train_path="data/test.hdf5", val_path="data/test.hdf5", num_workers=1
)

# %%
datamodule.setup("eval")

# %%
train_dataset = datamodule.train_dataset
# %%
sample = train_dataset[0]

# %%
import torch.nn as nn
import torch

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
