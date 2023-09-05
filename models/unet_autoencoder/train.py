import argparse

import torch
from datatime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.unet_autoencoder.dataset import AutoencoderDataModule
from models.unet_autoencoder.model import Autoencoder


def get_args():
    parser = argparse.ArgumentParser(
        "Autoencoder",
        add_help=False,
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        default="autoencoder",
        type=str,
        metavar="MODEL_NAME",
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
        "--batch_size", default=1, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers")
    parser.add_argument(
        "--max_epochs", default=5000, type=int, help="Max epochs to train"
    )
    parser.add_argument(
        "--train_path",
        default="data/autoencoder/train.npy",
        type=str,
        help="Path to is the npy file for training",
    )
    parser.add_argument(
        "--val_path",
        default="data/autoencoder/val.npy",
        type=str,
        help="Path to is the npy file for validation",
    )

    return parser.parse_args()


def train(args):
    hour_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    model = Autoencoder(lr=args.lr)
    datamodule = AutoencoderDataModule(
        args.data_path, args.val_data_path, args.batch_size, args.num_workers
    )

    experiment_name = f"{args.model_name}_{hour_str}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=(
            "lightning_logs/autoencoder"
            + f"{args.model_name}/{experiment_name}_checkpoints"
        ),
        filename=(
            f"{experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}-{{f1:.2f}}"
            f"-{{val_mae:.2f}}-{{val_mse:.2f}}-{{val_ssim:.2f}}"
        ),
        monitor="val_loss",
        mode="min",
        save_top_k=5,
    )

    tensorboard_logger = TensorBoardLogger(
        "lightning_logs/autoencoder", name=args.model_name
    )

    print("Running trainer")
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )

    trainer.validate(model, datamodule=datamodule)

    # Train the model
    trainer.fit(
        model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    args = get_args()
    train(args=args)
