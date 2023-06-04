import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class ADNIDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        transform=None,
        num_classes=3,
    ):
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.X = X
        self.y = y
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx] >= 0.5
        label = torch.LongTensor([label])

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label": label}
        return sample


class ADNIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        train_h5_ = h5py.File(self.train_path, "r")
        val_h5_ = h5py.File(self.val_path, "r")

        X_train, y_train = train_h5_["X"], train_h5_["y"]
        X_val, y_val = val_h5_["X"], val_h5_["y"]

        mean, std = mean_and_standard_deviation(X_train)

        train_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )  # TODO: Add augmentation
        val_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )  # TODO: More transforms?

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
            shuffle=True,
            pin_memory=True,
        )


def mean_and_standard_deviation(array):
    """Compute the mean and standard deviation of an array of images"""
    N, num_channels, _, _ = array.shape
    reshaped_array = np.reshape(array, (N, num_channels, -1))

    mean_values = np.mean(reshaped_array, axis=2)
    std_values = np.std(reshaped_array, axis=2)

    return mean_values, std_values
