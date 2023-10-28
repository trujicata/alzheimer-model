import argparse
import sys
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append("./")
from dataset import ADNIDataModule

from models.classifier3D.model import Classifier3D


def get_args():
    parser = argparse.ArgumentParser(
        "Alzheimer Disease Classifier for ADNI Dataset",
        add_help=False,
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        default="convnet3d",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--freeze_block",
        default=2,
        type=int,
        help="Block number to freeze",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
    parser.add_argument(
        "--max_epochs", default=5000, type=int, help="Max epochs to train"
    )
    parser.add_argument(
        "--data_path",
        default="data",
        type=str,
        help="Path to where we will store the h5 files",
    )

    return parser.parse_args()


def train(args):
    model_name = args.model_name
    hour_str = datetime.now().strftime("%m_%d_%H_%M")
    experiment_name = f"{model_name}_{hour_str}"

    print("Loading models")
    model = Classifier3D(lr=args.lr, name=model_name)

    print("Loading data module")
    datamodule = ADNIDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Callbacks
    print("Defining callbacks")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_logs/checkpoints/{experiment_name}",
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_f1:.2f}}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,
    )

    # Instantiate the TensorBoard logger
    tensorboard_logger = TensorBoardLogger("lightning_logs/classifier", name=model_name)

    # Set up the PyTorch Lightning trainer
    print("Running trainer")
    torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="32",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(
        model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    args = get_args()
    train(args)
