# %%
import research.start
import yaml
import torch
from models.vit_age_gender.model import ViTClassifier3D as ViTAGClassifier3D
from models.vanilla_vit.model import ViTClassifier3D
from models.resnet.model import Classifier3D as ResNetClassifier3D
from models.densenet.model import Classifier3D as DenseNetClassifier3D

# %%
## RESNET

# First, check the config file
config_path = "models/resnet/versions/config.yaml"
with open(config_path, "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

model = ResNetClassifier3D(
    lr=float(config["lr"]),
    model_depth=config["model_depth"],
    scheduler_step_size=config["scheduler_step_size"],
    scheduler_gamma=config["scheduler_gamma"],
    weight_decay=float(config["weight_decay"]),
    optimizer_alg=config["optimizer_alg"],
    name="test",
    class_weights=config["class_weights"],
    freeze=config["freeze"],
)

# %%
# Get the checkpoint
ckpt_path = "lightning_logs/checkpoints/resnet/resnet/resnet-P01-epoch=81-val_loss=0.65-val_recall=0.74.ckpt"

checkpoint_weights = torch.load(ckpt_path)["state_dict"]

# %%
# Check if the weights suit the model
model.load_state_dict(checkpoint_weights)
# %%
## DENSENET

# First, check the config file
config_path = "models/densenet/versions/config.yaml"
with open(config_path, "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

model = DenseNetClassifier3D(
    lr=float(config["lr"]),
    scheduler_step_size=config["scheduler_step_size"],
    scheduler_gamma=config["scheduler_gamma"],
    weight_decay=float(config["weight_decay"]),
    optimizer_alg=config["optimizer_alg"],
    name="test",
    class_weights=config["class_weights"],
    pretrained_backbone=config["pretrained_backbone"],
)
# %%
# Get the checkpoint
ckpt_path = "lightning_logs/checkpoints/densenet/densenet-P01-epoch=88-val_loss=0.45-val_f1=0.53-val_recall=0.59.ckpt"

checkpoint_weights = torch.load(ckpt_path)["state_dict"]
# %%
# Check if the weights suit the model
model.load_state_dict(checkpoint_weights)

# %%
## VIT

# First, check the config file
with open("models/vanilla_vit/versions/config.yaml", "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

model = ViTClassifier3D(
    name=config["model_name"],
    lr=float(config["lr"]),
    image_patch_size=config["image_patch_size"],
    frame_patch_size=config["frame_patch_size"],
    dim=config["dim"],
    depth=config["depth"],
    heads=config["heads"],
    mlp_dim=config["mlp_dim"],
    dim_head=config["dim_head"],
    pool=config["pool"],
    dropout=float(config["dropout"]),
    emb_dropout=float(config["emb_dropout"]),
    class_weights=config["class_weights"],
    weight_decay=float(config["weight_decay"]),
    scheduler_step_size=config["scheduler_step_size"],
    scheduler_gamma=config["scheduler_gamma"],
    optimizer_alg=config["optimizer_alg"],
)

# %%
# Get the checkpoint
ckpt_path = "lightning_logs/checkpoints/vit-vanilla/vit-vanilla-P01-epoch=42-val_loss=0.41-val_recall=0.71.ckpt"

checkpoint_weights = torch.load(ckpt_path)["state_dict"]
# %%
# Check if the weights suit the model
model.load_state_dict(checkpoint_weights)
# %%
## VIT AGE GENDER

# First, check the config file
with open("models/vit_age_gender/versions/config.yaml", "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

model = ViTAGClassifier3D(
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
# Get the checkpoint
ckpt_path = "lightning_logs/checkpoints/vit-age-gender/vit-age-gender-P01-epoch=25-val_loss=0.41-val_recall=0.74-val_ad_c_accuracy=0.94-val_mci_c_accuracy=0.84.ckpt"

checkpoint_weights = torch.load(ckpt_path)["state_dict"]

# %%
# Check if the weights suit the model
model.load_state_dict(checkpoint_weights)
# %%
