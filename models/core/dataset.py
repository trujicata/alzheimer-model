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
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        files = ["train.hdf5", "test.hdf5", "post_pet_diag.hdf5"]
        s3 = boto3.client("s3")
        for file_name in files:
            path = os.path.join(self.data_path, file_name)
            if not os.path.exists(path):
                print("Downloadin the data from s3")
                with open(path, "wb") as f:
                    s3.download_fileobj("normal-h5s", file_name, f)
        train_h5_ = h5py.File(os.path.join(self.data_path, "train.hdf5"), "r")
        val_1_h5_ = h5py.File(os.path.join(self.data_path, "test.hdf5"), "r")
        val_2_h5_ = h5py.File(os.path.join(self.data_path, "post_pet_diag.hdf5"), "r")

        # Get 50 different random samples from val_2_h5_ and add them to the train set

        indices = np.sort(
            np.random.choice(val_2_h5_["X_nii"].shape[0], 50, replace=False)
        )
        X_2_test, y_2_test = val_2_h5_["X_nii"][indices], val_2_h5_["y"][indices]
        X_train = np.concatenate((train_h5_["X_nii"], X_2_test))
        y_train = np.concatenate((train_h5_["y"], y_2_test))

        X_val = np.concatenate((val_1_h5_["X_nii"], val_2_h5_["X_nii"]))
        y_val = np.concatenate((val_1_h5_["y"], val_2_h5_["y"]))

        # mean, std = mean_and_standard_deviation(X_train)
        train_transforms = T.Compose([T.ToTensor()])  # TODO: Add augmentation
        val_transforms = T.Compose([T.ToTensor()])  # TODO: More transforms?

        self.train_dataset = ADNIDataset(X_train, y_train, transform=train_transforms)
        self.val_dataset = ADNIDataset(X_val, y_val, transform=val_transforms)
        self.test_dataset = ADNIDataset(X_2_test, y_2_test, transform=val_transforms)

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

    def check_balance(self, dataset: Dataset) -> None:
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            labels.append(torch.argmax(sample["label"]))
        labels = torch.stack(labels)
        print("Label counts: ", torch.unique(labels, return_counts=True))
        print(
            "Label counts: ", torch.unique(labels, return_counts=True)[1] / len(labels)
        )
