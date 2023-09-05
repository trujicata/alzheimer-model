import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class AutoencoderDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = self.load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self, data_path):
        data_npy = np.load(data_path)
        data_tensor = torch.from_numpy(data_npy)
        return data_tensor


class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, batch_size, num_workers):
        super().__init__()
        self.train_data = train_data_path
        self.val_data = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = AutoencoderDataset(self.data_path)
        self.val_dataset = AutoencoderDataset(self.val_data_path)

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
