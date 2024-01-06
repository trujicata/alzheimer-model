# %%
import start
import torch
import h5py
from models.classifier3D.model import Classifier3D

# %%
model = Classifier3D()

# weights = torch.load(
#     "lightning_logs/checkpoints/convnet3d-new-data/convnet3d-new-data-epoch=46-val_loss=0.45-val_f1=0.84.ckpt",
# )["state_dict"]

# for k, v in weights.items():
#     if k in model.state_dict().keys():
#         model.state_dict()[k].copy_(v)
#         print(f"Copied {k} from checkpoint to model")
#     else:
#         print(f"Key {k} not found in model state dict")
# %%
model.eval()
# %%
h5_train_path = "data/new-norm/test.hdf5"

h5_train = h5py.File(h5_train_path, "r")
train_data = h5_train["X_nii"]
labels = h5_train["y"]

# %%
from tqdm import tqdm

predictions_train = {"y_pred": [], "y_true": []}

for i in tqdm(range(len(train_data))):
    input = train_data[i]
    input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(input)
        pred = pred.argmax().item()
    predictions_train["y_pred"].append(pred)
    predictions_train["y_true"].append(int(labels[i]))

predictions_train
# %%
import matplotlib.pyplot as plt

plt.hist(predictions_train["y_pred"], bins=3)
plt.show()
# %%
import numpy as np

np.unique(predictions_train["y_pred"], return_counts=True)

# %%
plt.hist(predictions_train["y_true"], bins=3)
plt.show()
# %%
# Create a confusion matrix

from sklearn.metrics import confusion_matrix

m = confusion_matrix(
    predictions_train["y_true"], predictions_train["y_pred"], labels=[0, 1, 2]
)

m
# %%
plt.imshow(m)
plt.colorbar()
plt.show()
# %%
# calculate precision, recall and f1 score
from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(
    predictions_train["y_true"], predictions_train["y_pred"], average="weighted"
)
# %%
import os

best_result = {"ckpt": "", "f1": 0}

for ckpt in os.listdir("lightning_logs/checkpoints/convnet3d-new-data"):
    model = Classifier3D()
    weights = torch.load(
        f"lightning_logs/checkpoints/convnet3d-new-data/{ckpt}",
    )["state_dict"]

    for k, v in weights.items():
        if k in model.state_dict().keys():
            model.state_dict()[k].copy_(v)
            print(f"Copied {k} from checkpoint to model")
        else:
            print(f"Key {k} not found in model state dict")

    model.eval()

    predictions_train = {"y_pred": [], "y_true": []}

    for i in tqdm(range(len(train_data))):
        input = train_data[i]
        input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = model(input)
            pred = pred.argmax().item()
        predictions_train["y_pred"].append(pred)
        predictions_train["y_true"].append(int(labels[i]))

    pr, rec, f1 = precision_recall_fscore_support(
        predictions_train["y_true"], predictions_train["y_pred"], average="micro"
    )

    if f1 > best_result["f1"]:
        best_result["ckpt"] = ckpt
        best_result["f1"] = f1
