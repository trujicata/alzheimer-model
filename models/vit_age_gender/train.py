import argparse
import os
import sys

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append("./")
from models.vit_age_gender.dataset import ADNIDataModule
from models.vit_age_gender.model import ViTClassifier3D


def get_args():
    return parser.parse_args()


def get_args_from_yaml(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return argparse.Namespace(**config)


def train(args):
    model_name = args.model_name

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

    # Callbacks
    print("Defining callbacks")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_logs/checkpoints/{model_name}",
        filename=f"{model_name}-{args.processing}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_recall:.2f}}",
        monitor="val_recall",
        mode="max",
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Instantiate the TensorBoard logger
    tensorboard_logger = TensorBoardLogger(
        "lightning_logs/classifier/vit", name=model_name
    )

    config_copy_path = os.path.join(tensorboard_logger.log_dir, "config.yaml")

    # Set up the PyTorch Lightning trainer
    print("Running trainer")
    torch.set_float32_matmul_precision(args.precision)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="32",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, lr_monitor],
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
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
    except KeyboardInterrupt:
        print("Keyboard interrupt, saving config file")
        with open(config_copy_path, "w") as config_copy_file:
            yaml.dump(vars(args), config_copy_file)

    finally:
        with open(config_copy_path, "w") as config_copy_file:
            yaml.dump(vars(args), config_copy_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier model")
    parser.add_argument(
        "--config",
        metavar="config",
        type=str,
        help="Path to the YAML configuration file",
        default="models/vit_age_gender/versions/config.yaml",
    )
    args = parser.parse_args()
    args = get_args_from_yaml(args.config)
    train(args)
