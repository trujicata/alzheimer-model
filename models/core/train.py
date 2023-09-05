import argparse
import sys
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateFinder,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append("./")
from dataset import ADNIDataModule

from models.classifier3D.model import Classifier3D


def get_args():
    parser = argparse.ArgumentParser(
        " Alzheimer Disease Classifier for ADNI Dataset",
        add_help=False,
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        default="classifier3D",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
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
    model = Classifier3D(lr=args.lr)

    print("Loading data module")
    datamodule = ADNIDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Callbacks
    print("Defining callbacks")

    class FineTuneLearningRateFinder(LearningRateFinder):
        def __init__(self, milestones, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.milestones = milestones

        def on_fit_start(self, *args, **kwargs):
            return

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch in self.milestones:
                self.lr_find(trainer, pl_module)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_logs/watermark_detector/checkpoints/{experiment_name}",
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_f1:.2f}}",
        monitor="val_f1",
        mode="max",
        save_top_k=5,
    )

    ftlr_callback = FineTuneLearningRateFinder(
        milestones=[0, 5, 10, 20],
        min_lr=1e-8,
        max_lr=1e-1,
        num_training_steps=200,
        early_stop_threshold=8.0,
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

    # Instantiate the TensorBoard logger
    tensorboard_logger = TensorBoardLogger("lightning_logs/classifier", name=model_name)

    # Set up the PyTorch Lightning trainer
    print("Running trainer")
    torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="32",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, ftlr_callback, swa_callback],
        logger=tensorboard_logger,
    )

    # Train the model
    trainer.fit(
        model,
        datamodule=datamodule,
    )

    # Test the model
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    args = get_args()
    train(args)
