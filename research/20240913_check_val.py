# %%
import start
import argparse
import os
import sys

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.vit_age_gender.dataset import ADNIDataModule
from models.vit_age_gender.model import ViTClassifier3D


# %%
def get_args_from_yaml(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return argparse.Namespace(**config)


config = "models/vit_age_gender/versions/P01/config_v2.yaml"
args = get_args_from_yaml(config)


# %%
def val(args):
    # Set seed for reproducibility
    torch.manual_seed(42)

    print("Loading models")
    model = ViTClassifier3D(
        name=args.model_name,
        lr=float(args.lr),
        image_patch_size=args.image_patch_size,
        frame_patch_size=args.frame_patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        pool=args.pool,
        dropout=float(args.dropout),
        emb_dropout=float(args.emb_dropout),
        class_weights=args.class_weights,
        weight_decay=float(args.weight_decay),
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        optimizer_alg=args.optimizer_alg,
    )

    if args.checkpoint_path is not None:
        weights = torch.load(args.checkpoint_path)["state_dict"]
        model.load_state_dict(weights)

    print("Loading data module")
    datamodule = ADNIDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_cudim=args.include_cudim,
        processing=args.processing,
    )

    # Set up the PyTorch Lightning trainer
    print("Running trainer")
    torch.set_float32_matmul_precision(args.precision)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="32",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=5,
        gradient_clip_val=0.5,
    )

    # Validate the model in validation mode
    datamodule.setup(stage="validate")

    trainer.test(
        model, dataloaders=datamodule.val_dataloader(), ckpt_path=args.checkpoint_path
    )


# %%
val(args)

# %%
