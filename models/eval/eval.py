import torch
from models.eval.dataset import ClassificationDataModule as EvaluationDataModule
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
    elif model_name == "RegressionResNet":
        from models.regression_resnet.model import Classifier3D as Model
    elif model_name == "AlexNet":
        from models.alexnet.model import Classifier3D as Model
    elif model_name == "RegressionAlexNet":
        from models.regression_alexnet.model import Classifier3D as Model
    elif model_name == "ViT":
        from models.vanilla_vit.model import ViTClassifier3D as Model
    elif model_name == "ViTAgeGender":
        from models.vit_age_gender.model import ViTClassifier3D as Model

    model = Model(name=model_name)
    model.model.eval()

    tensorboard_logger = TensorBoardLogger(
        f"lightning_logs/evaluation/{model_name}/{checkpoint_path.split('/')[-1]}"
    )

    weights = torch.load(checkpoint_path)["state_dict"]
    for k, v in model.state_dict().items():
        if k in weights.keys():
            model.state_dict()[k] = weights[k]
        else:
            print(f"Key {k} not found in checkpoint")

    print("Loading data module")
    datamodule = EvaluationDataModule(
        data_path=data_path,
        post_or_pre=post_or_pre,
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Evaluate the model
    print("Evaluating the model")
    trainer = Trainer(
        logger=tensorboard_logger,
    )
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ResNet")
    parser.add_argument(
        "--data-path",
        default="data/new-norm",
        type=str,
    )
    parser.add_argument("--post-or-pre", default="post", type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="lightning_logs/checkpoints/resnet/resnet/resnet-epoch=16-val_loss=0.31-val_f1=0.81.ckpt",
    )
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        data_path=args.data_path,
        post_or_pre=args.post_or_pre,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint_path,
    )
