import torch
from models.eval.dataset import ClassificationDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


import argparse


def get_args(parser):
    parser.parse_args()


def main(
    model_name: str,
    data_path: str,
    post_or_pre: str,
    batch_size: int,
    num_workers: int,
    checkpoint_path: str,
):
    # Set seed for reproducibility
    torch.manual_seed(42)

    print("Loading models")
    if model_name == "ResNet":
        from models.resnet.model import Classifier3D as Model
    elif model_name == "RegressionResnet":
        from models.regression_resnet.model import Classifier3D as Model
    elif model_name == "AlexNet":
        from models.alexnet.model import Classifier3D as Model
    elif model_name == "RegressionAlexNet":
        from models.regression_alexnet.model import Classifier3D as Model
    elif model_name == "ViT":
        from models.vit.model import ViTClassifier3D as Model
    elif model_name == "ViTAgeGender":
        from models.vit.model import ViTClassifier3D as Model

    model = Model()

    tensorboard_logger = TensorBoardLogger(
        f"lightning_logs/evaluation/{model_name}/{checkpoint_path.split('/')[-1]}"
    )

    if checkpoint_path is not None:
        weights = torch.load(checkpoint_path)["state_dict"]
        model.load_state_dict(weights)

    print("Loading data module")
    datamodule = ClassificationDataModule(
        data_path=data_path,
        post_or_pre=post_or_pre,
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Evaluate the model
    print("Evaluating the model")
    trainer = Trainer()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--post_or_pre", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        data_path=args.data_path,
        post_or_pre=args.post_or_pre,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint_path,
    )
