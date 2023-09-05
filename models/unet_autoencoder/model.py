import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    StructuralSimilarityIndexMeasure,
)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(1024, 2048, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(2048, 2048, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(
            2048, 1024, kernel_size=4, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose3d(
            1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose3d(
            512, 512, kernel_size=(6, 4, 4), stride=2, padding=2, output_padding=1
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class Autoencoder(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.lr = lr

        self.criterion = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch  # [batch_size, 512, 13, 11, 11]
        x_hat = self(x)  # [batch_size, 512, 13, 11, 11]
        loss = self.criterion(x_hat, x)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
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
