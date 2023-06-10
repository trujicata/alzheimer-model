import torch
import torch.nn as nn
import pytorch_lightning as pl


class Classifier3D(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        num_classes: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential(
            self._conv_layer_set(1, 32),
            self._conv_layer_set(32, 64),
            nn.Flatten(),
            nn.Linear(12544, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=(3, 3, 3),
                stride=(3, 3, 3),
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

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
