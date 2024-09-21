import os

import boto3
import numpy as np
import h5py
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


def age_range(age):
    if age <= 50:
        return 0
    elif age < 50 and age <= 60:
        return 1
    elif age < 60 and age <= 70:
        return 2
    elif age < 70 and age <= 80:
        return 3
    elif age < 80 and age <= 90:
        return 4
    else:
        return 5


class ADNIDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        age,
        sex,
        transform=None,
        num_classes=3,
    ):
        self.X = X
        self.age = age
        self.sex = sex
        self.y = y

        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        age = age_range(self.age[idx])
        age = torch.tensor(age, dtype=torch.int64)
        sex = self.sex[idx]
        sex = torch.tensor(sex, dtype=torch.int64)

        label_id = int(self.y[idx])
        label = torch.zeros(self.num_classes)
        label[label_id] = 1

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": image.unsqueeze(0),
            "age": age.unsqueeze(0),
            "sex": sex.unsqueeze(0),
            "label": label,
        }
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

        val_1_h5_ = h5py.File(os.path.join(self.data_path, files[1]), "r")
        val_2_h5_ = h5py.File(os.path.join(self.data_path, files[2]), "r")
        train_h5_ = h5py.File(os.path.join(self.data_path, files[0]), "r")

        X_val, y_val = val_1_h5_["X_nii"], val_1_h5_["y"]
        X_test, y_test = val_2_h5_["X_nii"], val_2_h5_["y"]

        X_train, age_train, sex_train, y_train = (
            train_h5_["X_nii"],
            train_h5_["X_Age"],
            train_h5_["X_Sex"],
            train_h5_["y"],
        )
        X_val, age_val, sex_val, y_val = (
            val_1_h5_["X_nii"],
            val_1_h5_["X_Age"],
            val_1_h5_["X_Sex"],
            val_1_h5_["y"],
        )

        X_test, age_test, sex_test, y_test = (
            val_2_h5_["X_nii"],
            val_2_h5_["X_Age"],
            val_2_h5_["X_Sex"],
            val_2_h5_["y"],
        )

        if self.include_cudim:
            indices = np.sort(np.random.choice(X_test.shape[0], 50, replace=False))
            X_add, age_add, sex_add, y_add = (
                X_test[indices],
                age_test[indices],
                sex_test[indices],
                y_test[indices],
            )
            X_train = np.concatenate((X_train, X_add))
            age_train = np.concatenate((age_train, age_add))
            sex_train = np.concatenate((sex_train, sex_add))
            y_train = np.concatenate((y_train, y_add))

            X_val = np.concatenate((X_val, X_test))
            age_val = np.concatenate((age_val, age_test))
            sex_val = np.concatenate((sex_val, sex_test))
            y_val = np.concatenate((y_val, y_test))

        train_transforms = T.Compose([T.ToTensor()])  # TODO: Add augmentation
        val_transforms = T.Compose([T.ToTensor()])  # TODO: More transforms?

        self.train_dataset = ADNIDataset(
            X=X_train,
            age=age_train,
            sex=sex_train,
            y=y_train,
            transform=train_transforms,
        )
        self.val_dataset = ADNIDataset(
            X=X_val, age=age_val, sex=sex_val, y=y_val, transform=val_transforms
        )
        self.test_dataset = ADNIDataset(
            X=X_test,
            age=age_test,
            sex=sex_test,
            y=y_test,
            transform=val_transforms,
        )

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
