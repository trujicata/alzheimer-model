import os

import boto3
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class ADNIDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        transform=None,
        num_classes=3,
    ):
        self.X = X
        self.y = y
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label_id = int(self.y[idx])
        label = torch.zeros(self.num_classes)
        label[label_id] = 1

        if self.transform:
            image = self.transform(image)

        sample = {"image": image.unsqueeze(0), "label": label}
        return sample


class ADNIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        processing: str,
        batch_size: int = 32,
        num_workers: int = 4,
        include_cudim: bool = False,
    ):
        super().__init__()
        self.data_path = os.path.join(data_path, processing)
        self.processing = processing
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.include_cudim = include_cudim

    def setup(self, stage: str):
        files = [
            "train_csv.hdf5",
            "test_csv.hdf5",
            "pre_pet_diag.hdf5",
            "description.txt",
        ]
        s3 = boto3.client("s3")

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        for file_name in files:
            path = os.path.join(self.data_path, file_name)
            if not os.path.exists(path):
                print("Downloading the data from s3")
                key = f"{self.processing}/{file_name}"
                with open(path, "wb") as f:
                    s3.download_fileobj("brainers-preprocessed", key, f)

        train_h5_ = h5py.File(os.path.join(self.data_path, files[0]), "r")
        val_1_h5_ = h5py.File(os.path.join(self.data_path, files[1]), "r")
        val_2_h5_ = h5py.File(os.path.join(self.data_path, files[2]), "r")
        X_train, y_train = train_h5_["X_nii"], train_h5_["y"]

        X_val, y_val = val_1_h5_["X_nii"], val_1_h5_["y"]
        X_test, y_test = val_2_h5_["X_nii"], val_2_h5_["y"]

        if self.include_cudim:
            indices = np.sort(np.random.choice(X_test.shape[0], 50, replace=False))
            X_add, y_add = X_test[indices], y_test[indices]
            X_train = np.concatenate((X_train, X_add))
            y_train = np.concatenate((y_train, y_add))

            X_val = np.concatenate((X_val, X_test))
            y_val = np.concatenate((y_val, y_test))

        train_transforms = T.Compose([T.ToTensor()])  # TODO: Add augmentation
        val_transforms = T.Compose([T.ToTensor()])  # TODO: More transforms?

        self.train_dataset = ADNIDataset(X_train, y_train, transform=train_transforms)
        self.val_dataset = ADNIDataset(X_val, y_val, transform=val_transforms)
        self.test_dataset = ADNIDataset(X_test, y_test, transform=val_transforms)

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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
