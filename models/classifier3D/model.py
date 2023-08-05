from typing import List

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math


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
            layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(layer_out)

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv3d(
                layer_in, layer_out, kernel_size=1, stride=1, padding=0, bias=False
            )
        )

        self.act2 = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Classifier3D(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        num_classes: int = 3,
        input_size: List[int] = None,
    ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential([self._make_block(i) for i in range(5)])

        d, h, w = self._maxpool_output_size(input_size[1::], nb_layers=5)

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(128 * d * h * w, 256),
            nn.ELU(),
            nn.Dropout(p=0.8),
            nn.Linear(256, 2),
        )

    def _make_block(self, block_number, input_size=None):
        return nn.Sequential(
            ResBlock(block_number, input_size), nn.MaxPool3d(3, stride=2)
        )

    def _maxpool_output_size(
        self, input_size, kernel_size=(3, 3, 3), stride=(2, 2, 2), nb_layers=1
    ):
        d = math.floor((input_size[0] - kernel_size[0]) / stride[0] + 1)
        h = math.floor((input_size[1] - kernel_size[1]) / stride[1] + 1)
        w = math.floor((input_size[2] - kernel_size[2]) / stride[2] + 1)

        if nb_layers == 1:
            return d, h, w
        return self._maxpool_output_size(
            (d, h, w), kernel_size=kernel_size, stride=stride, nb_layers=nb_layers - 1
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
