from typing import List, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl


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
        num_classes: int = 3,
        input_size: List[int] = [181, 181, 217],
        depth: Optional[int] = 5,
        lr: Optional[float] = 0.001,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        h = int((input_size[0] - 3) / (2 ** (depth + 1))) + 2
        w = int((input_size[1] - 3) / (2 ** (depth + 1))) + 2
        d = int((input_size[2] - 3) / (2 ** (depth + 1))) + 2
        self.feature_extractor = nn.Sequential(
            *[self._make_block(i, 1) for i in range(depth)]
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(h*w*d, 256),
            nn.ELU(),
            nn.Dropout(p=0.8),
            nn.Linear(256, num_classes),
        )

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
