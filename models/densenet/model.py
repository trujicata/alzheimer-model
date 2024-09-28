import monai
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lion_pytorch import Lion
from matplotlib import pyplot as plt


class Classifier3D(pl.LightningModule):

    def __init__(
        self,
        lr: float = 0.001,
        scheduler_gamma: Optional[float] = 0.1,
        scheduler_step_size: Optional[int] = 25,
        weight_decay: float = 0.0,
        optimizer_alg: str = "adam",
        class_weights: Optional[list] = None,
        name: Optional[str] = None,
        freeze_backbone: bool = False,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.name = name
        self.lr = lr
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.weight_decay = weight_decay
        self.optimizer_alg = optimizer_alg
        self.num_classes = 3
        self.classes = ["AD", "MCI", "CN"]

        if class_weights is not None:
            class_weights = torch.Tensor(class_weights)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.model = monai.networks.nets.DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=3
        )
        if pretrained_backbone:
            checkpoint = torch.load("data/model3.pth")
            self.model.load_state_dict(checkpoint)
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        self.val_conf_matrix = ConfusionMatrixPloter(classes=self.classes)
        self.train_conf_matrix = ConfusionMatrixPloter(classes=self.classes)
        self.test_conf_matrix = ConfusionMatrixPloter(classes=self.classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer_alg == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_alg == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_alg == "lion":
            optimizer = Lion(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        if self.scheduler_gamma is not None and self.scheduler_step_size is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        class_predictions = logits.argmax(dim=1)
        preds = torch.zeros_like(logits)
        preds[torch.arange(logits.shape[0]), class_predictions] = 1
        self.train_conf_matrix.update(preds, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        class_predictions = logits.argmax(dim=1)
        preds = torch.zeros_like(logits)
        preds[torch.arange(logits.shape[0]), class_predictions] = 1

        self.val_conf_matrix.update(preds, y)

        self.log_dict(
            {
                "val_loss": loss,
            },
        )
        self.log_images(x, y, preds)

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        class_predictions = logits.argmax(dim=1)
        preds = torch.zeros_like(logits)
        preds[torch.arange(logits.shape[0]), class_predictions] = 1

        self.val_conf_matrix.update(preds, y)

        self.log_dict(
            {
                "test_loss": loss,
            },
        )
        self.log_images(x, y, preds)

    def on_test_epoch_end(self):
        precision, recall, f1 = self.calculate_metrics(self.test_conf_matrix.compute())
        self.log_conf_matrix(mode="test")
        self.log_dict(
            {
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
        )

    def on_validation_epoch_end(self) -> None:
        precision, recall, f1 = self.calculate_metrics(self.val_conf_matrix.compute())
        self.log_conf_matrix(mode="val")
        self.log_dict(
            {
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
            }
        )

    def on_train_epoch_end(self) -> None:
        precision, recall, f1 = self.calculate_metrics(self.train_conf_matrix.compute())
        self.log_conf_matrix(mode="train")
        self.log_dict(
            {
                "train_precision": precision,
                "train_recall": recall,
                "train_f1": f1,
            }
        )

    def log_conf_matrix(self, mode="val"):
        if mode == "val":
            fig = self.val_conf_matrix.plot()
            name = "Validation_Confusion_Matrix"
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
            self.val_conf_matrix.reset()
        elif mode == "test":
            fig = self.test_conf_matrix.plot()
            name = "Test_Confusion_Matrix"
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
            self.test_conf_matrix.reset()
        else:
            fig = self.train_conf_matrix.plot()
            name = "Train_Confusion_Matrix"
            self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
            self.train_conf_matrix.reset()
        plt.close()

    def log_images(self, images, labels, preds):
        random_index = np.random.randint(0, len(images))
        image = images[random_index]
        label = self.classes[labels[random_index].argmax().item()]
        pred = self.classes[preds[random_index].argmax().item()]

        slice_vertical = image[0, :, :, 55].detach().cpu().numpy()
        slice_horizontal = image[0, 45, :, :].detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(slice_vertical, cmap="gray")
        ax1.set_title(f"Label: {label}, Pred: {pred}")
        ax2.imshow(slice_horizontal, cmap="gray")
        ax2.set_title(f"Label: {label}, Pred: {pred}")

        self.logger.experiment.add_figure(
            "Random_Slices",
            fig,
            self.current_epoch,
        )
        plt.close(fig)

    def calculate_metrics(self, confusion_matrix):
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)

        for metric in [precision, recall, f1]:
            metric[np.isnan(metric)] = 0
            # Transform to torch
            metric = torch.Tensor(metric)

        return precision.mean(), recall.mean(), f1.mean()


class ConfusionMatrixPloter:
    def __init__(self, classes):
        self.num_classes = len(classes)
        self.classes = classes
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, preds, targets):
        conf_matrix = self.confusion_matrix(
            preds.detach().cpu(), targets.detach().cpu()
        ).numpy()
        self.matrix += conf_matrix

    def compute(self):
        return self.matrix

    def plot(self):
        plt.figure(figsize=(10, 10))
        normalized_matrix = self.matrix / self.matrix.sum(axis=1, keepdims=True)

        plt.imshow(normalized_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")

        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(
                    j,
                    i,
                    round(normalized_matrix[i, j], 2),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=26,
                )
        return plt.gcf()

    def reset(self):
        self.matrix *= 0

    def confusion_matrix(self, preds, target):
        matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int)
        for p, t in zip(preds, target):
            pred_class = torch.argmax(p)
            target_class = torch.argmax(t)
            matrix[target_class][pred_class] += 1
        return matrix
