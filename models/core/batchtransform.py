import os

import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from models.unet_autoencoder.unet_encoder import UNet3DEncoder


def batch_transform(encoder: UNet3DEncoder, train_path, val_path):
    train_h5 = h5py.File(train_path, "r")
    train_data = train_h5["X_nii"]
    train_labels = train_h5["y"]
    val_h5 = h5py.File(val_path, "r")
    val_labels = val_h5["y"]
    val_data = val_h5["X_nii"]

    totensor = T.ToTensor()

    os.makedirs("data/autoencoder/embeddings", exist_ok=True)

    train_embeddings = []
    for sample in tqdm(train_data, "Training data"):
        tensor_sample = totensor(sample)
        embedding = encoder(tensor_sample.unsqueeze(0).unsqueeze(0))
        train_embeddings.append(embedding.squeeze(0))
    torch.save(train_embeddings, "data/autoencoder/embeddings/train.pt")
    
    train_labs = []
    for y in train_labels:
        train_labs.append(y)

    val_embeddings = []
    for sample in tqdm(val_data, "Validation data"):
        tensor_sample = totensor(sample)
        embedding = encoder(tensor_sample.unsqueeze(0).unsqueeze(0))
        val_embeddings.append(embedding.squeeze(0))
    
    val_labs = []
    for y in val_labels:
        val_labs.append(y)
    
    torch.save(val_embeddings, "data/autoencoder/embeddings/val.pt")

    writer = SummaryWriter(log_dir=".runs/embeddings")

    writer.add_embedding(train_embeddings, metadata=train_labs)
    writer.add_embedding(val_embeddings, metadata=val_labs)