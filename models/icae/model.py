import pytorch_lightning as pl
import torch
import torch.nn as nn
from lion_pytorch import Lion
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    StructuralSimilarityIndexMeasure,
)

conv_block_1 = nn.Sequential(
    nn.Conv3d(1, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=2, stride=2),
    nn.BatchNorm3d(5),
)

conv_block_2 = nn.Sequential(
    nn.Conv3d(5, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=2, stride=2),
    nn.BatchNorm3d(5),
)

conv_block_3 = nn.Sequential(
    nn.Conv3d(5, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=2, stride=2),
    nn.BatchNorm3d(5),
)

encoder = nn.Sequential(conv_block_1, conv_block_2, conv_block_3)

unconv_block_1 = nn.Sequential(
    nn.ConvTranspose3d(5, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.BatchNorm3d(5),
)

unconv_block_2 = nn.Sequential(
    nn.ConvTranspose3d(5, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.BatchNorm3d(5),
)

unconv_block_3 = nn.Sequential(
    nn.ConvTranspose3d(5, 1, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.BatchNorm3d(1),
)

decoder = nn.Sequential(unconv_block_1, unconv_block_2, unconv_block_3)


class ICAE(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        optimizer_alg: str = "adam",
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
        weight_decay: float = 0.0,
        name: str = "ICAE",
    ):
        super(ICAE, self).__init__()
        self.name = name
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.weight_decay = weight_decay
        self.optimizer_alg = optimizer_alg
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.criterion = nn.MSELoss()

        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        if self.optimizer_alg == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_alg == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_alg == "lion":
            optimizer = Lion(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
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
        x = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)

        self.mae(x_hat, x)
        self.mse(x_hat, x)
        self.ssim(x_hat, x)

        self.log_dict(
            {
                "val_loss": loss,
                "val_mae": self.mae,
                "val_mse": self.mse,
                "val_ssim": self.ssim,
            },
        )
