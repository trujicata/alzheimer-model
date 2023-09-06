#%%
import start
import torch

#%%
train_tensors = torch.load("data/autoencoder/embeddings/train.pt")
# %%
train_tensors.shape
# %%
train_tensors[0].shape
# %%
