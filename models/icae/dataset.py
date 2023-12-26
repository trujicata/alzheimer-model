import os

import boto3
import h5py
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class ADNIDataset(Dataset):
    def __init__(
        self,
        X,
        transform=None,
    ):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        image = image[2:-2, 2:-2, :]

        if self.transform:
            image = self.transform(image)

        return image.unsqueeze(0)


class ADNIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        files = ["train.hdf5", "test.hdf5", "holdout.hdf5"]
        s3 = boto3.client("s3")
        for file_name in files:
            path = os.path.join(self.data_path, file_name)
            if not os.path.exists(path):
                print("Downloadin the data from s3")
                with open(path, "wb") as f:
                    s3.download_fileobj("normal-h5s", file_name, f)
        train_h5_ = h5py.File(os.path.join(self.data_path, "train.hdf5"), "r")
        val_h5_ = h5py.File(os.path.join(self.data_path, "test.hdf5"), "r")
        holdout_h5_ = h5py.File(os.path.join(self.data_path, "holdout.hdf5"), "r")

        X_train = train_h5_["X_nii"]
        X_val = val_h5_["X_nii"]

        # mean, std = mean_and_standard_deviation(X_train)
        train_transforms = T.Compose([T.ToTensor()])  # TODO: Add augmentation
        val_transforms = T.Compose([T.ToTensor()])  # TODO: More transforms?

        self.train_dataset = ADNIDataset(X_train, transform=train_transforms)
        self.val_dataset = ADNIDataset(X_val, transform=val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
