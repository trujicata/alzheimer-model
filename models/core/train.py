import argparse
import os
import sys

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append("./")
from dataset import ADNIDataModule

from models.classifier3D.model import Classifier3D


def get_args_from_yaml(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return argparse.Namespace(**config)

def train(args):
    model_name = args.model_name

    print("Loading models")
    model = Classifier3D(lr=float(args.lr), name=model_name, class_weights=args.class_weights)

    print("Loading data module")
    datamodule = ADNIDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Callbacks
    print("Defining callbacks")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_logs/{model_name}/checkpoints",
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_f1:.2f}}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,
    )

    # Instantiate the TensorBoard logger
    tensorboard_logger = TensorBoardLogger("lightning_logs/classifier", name=model_name)

    config_copy_path = os.path.join(tensorboard_logger.log_dir, f"config.yaml")

    # Set up the PyTorch Lightning trainer
    print("Running trainer")
    torch.set_float32_matmul_precision(args.precision)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="32",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=5,
        gradient_clip_val=0.5,
    )

    # Train the model
    try:
        trainer.fit(
            model,
            datamodule=datamodule,
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt, saving config file")
        with open(config_copy_path, 'w') as config_copy_file:
            yaml.dump(vars(args), config_copy_file)
    
    finally:
        with open(config_copy_path, 'w') as config_copy_file:
            yaml.dump(vars(args), config_copy_file)


if __name__ == "__main__":
    args = get_args_from_yaml("models/core/versions/config.yaml")
    train(args)
