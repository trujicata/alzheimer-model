import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AutoencoderDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.from_numpy(sample).float()
        return sample


class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size, num_workers):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = AutoencoderDataset(self.train_path)
        self.val_dataset = AutoencoderDataset(self.val_path)

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
