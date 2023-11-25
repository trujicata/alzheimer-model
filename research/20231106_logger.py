#%%
import start

from pytorch_lightning.loggers import TensorBoardLogger

# %%
tensorboard_logger = TensorBoardLogger("lightning_logs/classifier", name="convnet3d")

# %%
tensorboard_logger.log_dir
# %%
import argparse
import yaml
from models.classifier3D.model import Classifier3D

def get_args_from_yaml(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return argparse.Namespace(**config)

args = get_args_from_yaml("models/core/versions/config.yaml")
#%%
model = Classifier3D(lr=args.lr, name="hola", class_weights=args.class_weights)
model.train()
# %%
import torch.nn as nn
import torch
for k,v in model.model.state_dict().items():
        if v.dtype != torch.float32:
            print(k, v.dtype)
# %%
