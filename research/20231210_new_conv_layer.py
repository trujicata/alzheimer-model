# %%
import start

import torch
import torch.nn as nn


# %%
class ConvNet(nn.Module):
    def __init__(self, num_blocks: int = 3, dropout: float = 0.01):
        super(ConvNet, self).__init__()

        self.conv_blocks = self.create_conv_blocks(num_blocks)

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(1260, 512)),
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(512, 256)),
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(256, 128)),
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(128, 64)),
            nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(64, 32)),
            nn.Sequential(
                nn.Dropout(dropout), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=1)
            ),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def create_conv_blocks(self, num_blocks: int = 3):
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                n = 1
            else:
                n = 5
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(n, 5, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=2, stride=2),
                    nn.BatchNorm3d(5),
                )
            )
        return nn.Sequential(*blocks)


# %%
conv = ConvNet(num_blocks=4)

# %%
random_img = torch.randn((1, 1, 100, 100, 120))
out = conv(random_img)

# %%
