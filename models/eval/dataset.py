import os
from typing import Literal

import boto3
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.regression_alexnet.dataset import ADNIDataset as ADNIDatasetRegression
from models.core.dataset import ADNIDataset as ADNIDatasetCore
from models.vit.dataset import ADNIDataset as ADNIDatasetVit
from models.vit_age_gender.dataset import ADNIDataset as ADNIDatasetVitAgeGender


class ClassificationDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_path: str,
        post_or_pre: Literal["pre", "post"],
        model_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_path = data_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.post_or_pre = post_or_pre

        if model_name == "ResNet" or model_name == "AlexNet":
            self.dataset = ADNIDatasetCore
        elif model_name == "RegressionResNet" or model_name == "RegressionAlexNet":
            self.dataset = ADNIDatasetRegression
        elif model_name == "ViT":
            self.dataset = ADNIDatasetVit
        elif model_name == "ViTAgeGender":
            self.dataset = ADNIDatasetVitAgeGender

    def setup(self, stage: str):
        file = f"{self.post_or_pre}_pet_diag.hdf5"
        path = os.path.join(self.data_path, file)
        if not os.path.exists(path):
            print("Downloading the data from s3")
            s3 = boto3.client("s3")
            with open(path, "wb") as f:
                s3.download_fileobj("normal-h5s", file, f)

        ds_h5_ = h5py.File(os.path.join(path), "r")

        X, y = ds_h5_["X_nii"], ds_h5_["y"]

        transforms = T.Compose([T.ToTensor()])  # TODO: Add augmentation

        self.test_dataset = self.dataset(X, y, transform=transforms)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
