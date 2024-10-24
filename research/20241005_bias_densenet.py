# %%
import research.start
from models.densenet.model import Classifier3D

from models.densenet.dataset import ADNIDataset
import pandas as pd
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm
import torchvision.transforms as T

from sklearn.metrics import f1_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# Load Test Data
data_path = "data"
processing = "P03"
test_h5_ = h5py.File(f"{data_path}/{processing}/pre_pet_diag.hdf5", "r")

X_test, age_test, sex_test, y_test = (
    test_h5_["X_nii"],
    test_h5_["X_Age"],
    test_h5_["X_Sex"],
    test_h5_["y"],
)
# %%
config_path = "models/densenet/versions/config3.yaml"
with open(config_path, "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

# %%
transforms = T.Compose([T.ToTensor()])
test_dataset = ADNIDataset(
    X=X_test,
    y=y_test,
    transform=transforms,
)

# %%
# Load model
import os

ckpt_folder_path = "lightning_logs/checkpoints/densenet/P03"

for file in tqdm(os.listdir(ckpt_folder_path), desc="Different Checkpoints"):
    if file.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_folder_path, file)
        ckpt_weights = torch.load(ckpt_path)["state_dict"]
    else:
        continue

    try:
        model = Classifier3D(
            lr=float(config["lr"]),
            scheduler_step_size=config["scheduler_step_size"],
            scheduler_gamma=config["scheduler_gamma"],
            weight_decay=float(config["weight_decay"]),
            optimizer_alg=config["optimizer_alg"],
            name="test",
            class_weights=config["class_weights"],
            pretrained_backbone=config["pretrained_backbone"],
        )
        model.load_state_dict(ckpt_weights)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Error loading {ckpt_path}")
        continue
    # Get the predictions in the whole test set and create a pandas dataframe

    test_df = pd.DataFrame(columns=["Age", "Sex", "Prediction", "Label"])

    for i in tqdm(range(len(test_dataset)), desc="Test Set"):
        X = test_dataset[i]["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(X)
            prediction = torch.argmax(output, dim=1).item()
        label = torch.argmax(test_dataset[i]["label"]).item()
        age = age_test[i]
        sex = sex_test[i]

        datapoint = {
            "Age": age,
            "Sex": sex,
            "Prediction": prediction,
            "Label": label,
        }

        test_df = test_df.append(datapoint, ignore_index=True)

        # Calculate the recall and F1 score for the whole test set

    y_true = test_df["Label"]
    y_pred = test_df["Prediction"]

    recall = recall_score(y_true, y_pred, average="weighted")
    recall = round(recall, 4)

    f1 = f1_score(y_true, y_pred, average="weighted")
    f1 = round(f1, 4)

    # Save the results in a csv file
    test_df.to_csv(f"data/results/densenet_{processing}_{recall}_{f1}_test_results.csv")


# %%
