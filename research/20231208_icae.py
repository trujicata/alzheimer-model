# %%
import start

import torch
import torch.nn as nn

# %%
conv_block_1 = nn.Sequential(
    nn.Conv3d(1, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=2, stride=2),
    nn.BatchNorm3d(5),
)

conv_block_2 = nn.Sequential(
    nn.Conv3d(5, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=2, stride=2),
    nn.BatchNorm3d(5),
)

conv_block_3 = nn.Sequential(
    nn.Conv3d(5, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=2, stride=2),
    nn.BatchNorm3d(5),
)

encoder = nn.Sequential(conv_block_1, conv_block_2, conv_block_3)

# %%
random_image = torch.rand(1, 1, 96, 96, 120)


# %%
enc_out = encoder(random_image)
enc_out.view(enc_out.size(0), -1).shape
# %%
# Recontructing the image

unconv_block_1 = nn.Sequential(
    nn.ConvTranspose3d(5, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.BatchNorm3d(5),
)

unconv_block_2 = nn.Sequential(
    nn.ConvTranspose3d(5, 5, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.BatchNorm3d(5),
)

unconv_block_3 = nn.Sequential(
    nn.ConvTranspose3d(5, 1, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.BatchNorm3d(1),
)

decoder = nn.Sequential(unconv_block_1, unconv_block_2, unconv_block_3)


# %%
dec_out = decoder(enc_out)
dec_out.shape
# %%
