import os

import boto3
import h5py
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


def class_trad(x):
    if x == 0:  # AD
        return 1
    elif x == 1:  # MCI
        return 0.5
    elif x == 2:  # CN
        return 0


def class_trad2(x, params):
    if x < params[0]:  # CN
        return 2
    elif x < params[1]:  # MCI
        return 1
    else:  # AD
        return 0


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
        label = class_trad(label_id)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0).transpose(0, -1)

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

        X_train, y_train = train_h5_["X_nii"], train_h5_["y"]
        X_val, y_val = val_h5_["X_nii"], val_h5_["y"]

        # mean, std = mean_and_standard_deviation(X_train)
        train_transforms = T.Compose([T.ToTensor()])  # TODO: Add augmentation
        val_transforms = T.Compose([T.ToTensor()])  # TODO: More transforms?

        self.train_dataset = ADNIDataset(X_train, y_train, transform=train_transforms)
        self.val_dataset = ADNIDataset(X_val, y_val, transform=val_transforms)

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
