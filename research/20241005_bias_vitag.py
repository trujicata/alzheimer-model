# %%
import research.start
from models.vit_age_gender.model import ViTClassifier3D
from models.vit_age_gender.dataset import ADNIDataset
import torchvision.transforms as T

import yaml
import pandas as pd
import h5py
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score


# Make everything non verbose for the sake of the notebook
import logging

logging.disable(logging.CRITICAL)
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
config_path = "models/vit_age_gender/versions/config3.yaml"
with open(config_path, "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

# %%
model = ViTClassifier3D(
    name="test",
    lr=float(config["lr"]),
    image_patch_size=config["image_patch_size"],
    frame_patch_size=config["frame_patch_size"],
    dim=config["dim"],
    depth=config["depth"],
    heads=config["heads"],
    mlp_dim=config["mlp_dim"],
    pool=config["pool"],
    age_classes=config["age_classes"],
    dropout=float(config["dropout"]),
    emb_dropout=float(config["emb_dropout"]),
    class_weights=config["class_weights"],
    weight_decay=float(config["weight_decay"]),
    scheduler_step_size=config["scheduler_step_size"],
    scheduler_gamma=config["scheduler_gamma"],
    optimizer_alg=config["optimizer_alg"],
)

# %%
# Create the dataset
transforms = T.Compose([T.ToTensor()])
test_dataset = ADNIDataset(
    X=X_test,
    y=y_test,
    age=age_test,
    sex=sex_test,
    transform=transforms,
    age_classes=config["age_classes"],
)


# %%
# Load model
import os

ckpt_folder_path = "lightning_logs/checkpoints/vit-age-gender/P03"

for file in tqdm(os.listdir(ckpt_folder_path), desc="Different Checkpoints"):
    if file.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_folder_path, file)
        checkpoint_weights = torch.load(ckpt_path)["state_dict"]
    else:
        continue

    try:
        model = ViTClassifier3D(
            name="test",
            lr=float(config["lr"]),
            image_patch_size=config["image_patch_size"],
            frame_patch_size=config["frame_patch_size"],
            dim=config["dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"],
            pool=config["pool"],
            age_classes=config["age_classes"],
            dropout=float(config["dropout"]),
            emb_dropout=float(config["emb_dropout"]),
            class_weights=config["class_weights"],
            weight_decay=float(config["weight_decay"]),
            scheduler_step_size=config["scheduler_step_size"],
            scheduler_gamma=config["scheduler_gamma"],
            optimizer_alg=config["optimizer_alg"],
        )
        model.load_state_dict(checkpoint_weights)
        model.eval()
        model.to(device)
    except:
        print(f"Error loading {ckpt_path}")
        continue

    # Get the predictions in the whole test set and create a pandas dataframe

    test_df = pd.DataFrame(columns=["Age", "Sex", "Prediction", "Label"])

    for i in tqdm(range(len(test_dataset)), desc="Test Set"):
        image = test_dataset[i]["image"].unsqueeze(0).to(device)
        sex = test_dataset[i]["sex"].unsqueeze(0).to(device)
        age = test_dataset[i]["age"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image, age, sex)
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

    recall = recall_score(y_true, y_pred, average="micro")
    recall = round(recall, 4)

    f1 = f1_score(y_true, y_pred, average="micro")
    f1 = round(f1, 4)

    # Save the results in a csv file
    test_df.to_csv(f"data/results/vit_ag_{processing}_{recall}_{f1}_test_results.csv")

# %%

# %%
