# %%
import start

import torch
from models.classifier3D.model import ResBlock, Flatten
import torch.nn as nn


# %%
class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()

        # Initial convolutional layer
        self.conv_initial = nn.Conv3d(
            1, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_initial = nn.BatchNorm3d(16)
        self.act_initial = nn.ELU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResBlock(1, 16),
            ResBlock(2, 16),
            ResBlock(3, 16),
            # Add more ResBlock instances as needed
        )

        # Final convolutional layer
        self.conv_final = nn.Conv3d(
            16, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_final = nn.BatchNorm3d(1)
        self.act_final = nn.ELU()

    def forward(self, x):
        # Initial convolutional layer
        out = self.conv_initial(x)
        out = self.bn_initial(out)
        out = self.act_initial(out)

        # Residual blocks
        out = self.res_blocks(out)

        # Final convolutional layer
        out = self.conv_final(out)
        out = self.bn_final(out)
        out = self.act_final(out)

        return out


# Create an instance of the ResidualNet
model = ResidualNet()

# Print the model architecture
print(model)
# %%
image = torch.randn(1, 1, 100, 100, 120)
output = model(image)

# %%
model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=False)
model.eval()
# %%
