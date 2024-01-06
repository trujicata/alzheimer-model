# %%
import os

import h5py
import start
import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from models.classifier3D.model import Classifier3D

# %%
h5_train_path = "data/new-norm/test.hdf5"

h5_train = h5py.File(h5_train_path, "r")
train_data = h5_train["X_nii"]
labels = h5_train["y"]
# %%
best_result = {"ckpt": "", "f1": 0}

for ckpt in tqdm(os.listdir("lightning_logs/checkpoints/convnet3d-new-data")):
    model = Classifier3D()
    weights = torch.load(
        f"lightning_logs/checkpoints/convnet3d-new-data/{ckpt}",
    )["state_dict"]

    for k, v in weights.items():
        if k in model.state_dict().keys():
            model.state_dict()[k].copy_(v)

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

    _, _, f1, _ = precision_recall_fscore_support(
        predictions_train["y_true"], predictions_train["y_pred"], average="micro"
    )

    if f1 > best_result["f1"]:
        print(f"New best model found! F1: {f1}")
        best_result["ckpt"] = ckpt
        best_result["f1"] = f1

    del model
    torch.cuda.empty_cache()

# %%
best_model_path = f"lightning_logs/checkpoints/convnet3d-new-data/{best_result['ckpt']}"
print(f"Best model found at {best_model_path} with {best_result['f1']}")
