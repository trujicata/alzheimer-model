#%%
import start
import torch
#%%
train_emb= torch.load("data/autoencoder/embeddings/train.pt")
# %%
train_emb.shape
# %%
import numpy as np
#%%

np_emb = train_emb.numpy()
# %%
np.save("data/autoencoder/embeddings/train.npy", np_emb)
# %%
val_emb = torch.load("data/autoencoder/embeddings/val.pt")
val_np = val_emb.numpy()
#%%
np.save("data/autoencoder/embeddings/val.npy", val_np)
# %%
