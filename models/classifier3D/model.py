from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lion_pytorch import Lion
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


class ConvNet(nn.Module):
    def __init__(self, dropout: float = 0.01):
        super(ConvNet, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(1, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(5),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(5),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(5),
        )
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(10800, 64)),
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(64, 32)),
            nn.Sequential(
                nn.Dropout(dropout), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=1)
            ),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class Classifier3D(pl.LightningModule):
    def __init__(
        self,
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

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model = ConvNet(dropout=dropout)

        if freeze_block is not None:
            self.freeze_linear(freeze_block)

        self.precision = Precision(task="multiclass", num_classes=self.num_classes)
        self.recall = Recall(task="multiclass", num_classes=self.num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        self.val_conf_matrix = ConfusionMatrixPloter(classes=self.classes)
        self.train_conf_matrix = ConfusionMatrixPloter(classes=self.classes)

    def freeze_linear(self, block_number):
        for param in self.model.classifier[block_number].parameters():
            param.requires_grad = False

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

        self.precision(preds, y)
        self.recall(preds, y)
        self.f1(preds, y)
        self.val_conf_matrix.update(preds, y)

        self.log_dict(
            {
                "val_loss": loss,
                "val_f1": self.f1,
                "val_precision": self.precision,
                "val_recall": self.recall,
            }
        )
        self.log_images(x, y, preds)

    def on_validation_epoch_end(self) -> None:
        self.log_conf_matrix(mode="val")

    def on_train_epoch_end(self) -> None:
        self.log_conf_matrix(mode="train")

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
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(
                    j,
                    i,
                    str(int(self.matrix[i, j])),
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
