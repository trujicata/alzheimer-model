from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lion_pytorch import Lion
from matplotlib import pyplot as plt

from models.core.dataset import class_trad2


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


class ConvNet(nn.Module):
    def __init__(self, num_blocks: int = 3, dropout: float = 0.01):
        super(ConvNet, self).__init__()

        self.conv_blocks = self.create_conv_blocks(num_blocks)

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(1260, 512)),
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(512, 64)),
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(64, 32)),
            nn.Sequential(
                nn.Dropout(dropout), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
            ),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def create_conv_blocks(self, num_blocks: int = 3):
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                n = 1
            else:
                n = 5
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(n, 5, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=2, stride=2),
                    nn.BatchNorm3d(5),
                )
            )
        return nn.Sequential(*blocks)


class Classifier3D(pl.LightningModule):
    def __init__(
        self,
        num_conv_blocks: int = 4,
        dropout: float = 0.01,
        freeze_block: Optional[int] = None,
        lr: float = 0.001,
        scheduler_gamma: Optional[float] = 0.1,
        scheduler_step_size: Optional[int] = 25,
        weight_decay: float = 0.0,
        optimizer_alg: str = "adam",
        class_weights: Optional[list] = None,
        name: Optional[str] = None,
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

        self.criterion = nn.MSELoss()

        self.model = ConvNet(num_blocks=num_conv_blocks, dropout=dropout)

        if freeze_block is not None:
            self.freeze_feature_extractor(freeze_block)

        self.val_conf_matrix = ConfusionMatrixPloter(classes=self.classes)
        self.train_conf_matrix = ConfusionMatrixPloter(classes=self.classes)

    def freeze_feature_extractor(self, block_number):
        for i in range(block_number):
            self.model.conv_blocks[i].requires_grad_(False)

    def forward(self, x):
        x = self.model(x)
        return x

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

        class_predictions = [class_trad2(x) for x in logits]
        preds = torch.zeros(logits.shape[0], 3)
        preds[torch.arange(logits.shape[0]), class_predictions] = 1

        class_targets = [class_trad2(x) for x in y]
        y = torch.zeros(y.shape[0], 3)
        y[torch.arange(y.shape[0]), class_targets] = 1

        self.train_conf_matrix.update(preds, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        class_predictions = [class_trad2(x) for x in logits]
        preds = torch.zeros(logits.shape[0], 3)
        preds[torch.arange(logits.shape[0]), class_predictions] = 1

        class_targets = [class_trad2(x) for x in y]
        y = torch.zeros(y.shape[0], 3)
        y[torch.arange(y.shape[0]), class_targets] = 1

        self.val_conf_matrix.update(preds, y)

        self.log_dict(
            {
                "val_loss": loss,
            },
        )
        self.log_images(x, y, preds)

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
