import start
import torch
import numpy as np
import os

# Define the paths to the tensors
train_tensor_path = "data/autoencoder/embeddings/train.pt"
val_tensor_path = "data/autoencoder/embeddings/val.pt"

# Load and save the train tensor
train_tensor = torch.load(train_tensor_path)
train_npy_path = os.path.splitext(train_tensor_path)[0] + ".npy"
np.save(train_npy_path, train_tensor.numpy())
del train_tensor  # Delete the tensor from memory
os.remove(train_tensor_path)


# Load and save the validation tensor
val_tensor = torch.load(val_tensor_path)
val_npy_path = os.path.splitext(val_tensor_path)[0] + ".npy"
np.save(val_npy_path, val_tensor.numpy())
del val_tensor  # Delete the tensor from memory

# Delete the original tensor files
os.remove(val_tensor_path)

print(
    f"Tensors saved as NumPy arrays and deleted from disk:\nTrain: {train_npy_path}\nValidation: {val_npy_path}"
)
