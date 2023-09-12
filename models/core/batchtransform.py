import os
import sys

import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

sys.path.append("./")

from models.unet_autoencoder.unet_encoder import UNet3DEncoder


class H5Dataset(Dataset):
    def __init__(self, path: str):
        super(H5Dataset, self).__init__()
        self.h5 = h5py.File(path, "r")
        self.data = self.h5["X_nii"]
        self.labels = self.h5["y"]
        self.totensor = T.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return self.totensor(sample).unsqueeze(0), label


def batch_transform(
    encoder: UNet3DEncoder,
    train_path: str,
    val_path: str,
    batch_size: int = 10,
    num_workers: int = 4,
    device="cuda:0",
):
    device = torch.device(device)
    train_data = H5Dataset(train_path)
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    train_labels = train_data.labels

    os.makedirs("data/autoencoder/embeddings", exist_ok=True)

    train_embeddings = []
    for x, _ in tqdm(train_data_loader):
        x = x.to(device)
        with torch.no_grad():
            out = encoder(x)
        train_embeddings.append(out.squeeze(0).detach().cpu())

    train_embeddings = torch.stack(train_embeddings, dim=0).squeeze(1)
    print(train_embeddings.shape)
    torch.save(train_embeddings, "data/autoencoder/embeddings/train.pt")

    train_labs = []
    for y in train_labels:
        train_labs.append(y)

    val_data = H5Dataset(val_path)
    val_data_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_labels = val_data.labels

    val_labs = []
    for y in val_labels:
        val_labs.append(y)

    val_embeddings = []
    for x, _ in tqdm(val_data_loader):
        x = x.to(device)
        with torch.no_grad():
            out = encoder(x)
        val_embeddings.append(out.squeeze(0).detach().cpu())

    val_embeddings = torch.stack(val_embeddings, dim=0).squeeze(1)
    print(val_embeddings.shape)
    torch.save(val_embeddings, "data/autoencoder/embeddings/val.pt")

    writer = SummaryWriter(log_dir=".runs/embeddings")
    writer.add_embedding(train_embeddings, metadata=train_labs)
    writer.add_embedding(val_embeddings, metadata=val_labs)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    num_workers = 4
    encoder = UNet3DEncoder()
    encoder.load_state_dict(torch.load("models/weights/Genesis_Chest_CT_encoder.pt"))
    encoder.eval()
    encoder.to(device)

    batch_transform(
        encoder,
        "data/train.hdf5",
        "data/test.hdf5",
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
