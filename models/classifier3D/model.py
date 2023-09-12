import random
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchmetrics import F1Score, Precision, Recall


class Flatten(nn.Module):
    """Flatten a tensor"""

    def forward(self, input):
        return input.view(input.size(0), -1)


class ResBlock(nn.Module):
    def __init__(self, block_number, input_size):
        """Residual block for 3D CNN"""
        super(ResBlock, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2)

        self.conv1 = nn.Conv3d(
            layer_in, layer_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv3d(
            layer_out, layer_in, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(layer_in)

        self.act2 = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.act2(out)
        return out


class Classifier3D(pl.LightningModule):
    def __init__(
        self,
        num_classes: Optional[int] = 3,
        input_size: List[int] = [181, 181, 217],
        depth: Optional[int] = 3,
        lr: Optional[float] = 0.001,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.classes = ["AD", "MCI", "CN"]

        self.criterion = nn.BCEWithLogitsLoss()

        self.feature_extractor = nn.Sequential(
            *[self._make_block(i, 1) for i in range(depth)]
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(1200, 256),
            nn.ELU(),
            nn.Dropout(p=0.8),
            nn.Linear(256, num_classes),
        )

        self.precision = Precision(task="multiclass", num_classes=self.num_classes)
        self.recall = Recall(task="multiclass", num_classes=self.num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        self.conf_matrix = ConfusionMatrixPloter(classes=self.classes)

    def _make_block(self, block_number, num_input_channels):
        return nn.Sequential(
            ResBlock(block_number, num_input_channels), nn.MaxPool3d(3, stride=2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.classifier(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        class_predictions = logits.argmax(dim=1)
        preds = torch.zeros_like(logits)
        preds[torch.arange(logits.shape[0]), class_predictions] = 1

        self.precision(preds, y)
        self.recall(preds, y)
        self.f1(preds, y)
        self.conf_matrix.update(preds, y)

        self.log_dict(
            {
                "val_loss": loss,
                "val_f1": self.f1,
                "val_precision": self.precision,
                "val_recall": self.recall,
            }
        )
        self.log_images(x, y, preds)


        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_conf_matrix()

    def log_conf_matrix(self):
        self.conf_matrix.plot()
        self.logger.experiment.add_figure(
            "Confusion_Matrix", plt.gcf(), global_step=self.current_epoch
        )
        self.conf_matrix.reset()

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

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")

        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

    def reset(self):
        self.matrix *= 0

    def confusion_matrix(self, preds, target):
        matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int)
        for p, t in zip(preds, target):
            pred_class = torch.argmax(p)
            target_class = torch.argmax(t)
            matrix[target_class][pred_class] += 1
        return matrix
