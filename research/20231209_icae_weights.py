# %%
import start

import torch

from models.classifier3D.model import Classifier3D
from models.icae.model import ICAE

# %%
icae_weights = torch.load(
    "lightning_logs/checkpoints/icae/icae-epoch=18-val_loss=0.11-val_ssim=0.88.ckpt"
)["state_dict"]
icae_weights

# %%
icae_model = ICAE()
icae_model.load_state_dict(icae_weights)
# %%
encoder = icae_model.encoder
encoder
# %%
classifier = Classifier3D()
classifier.model
# %%
new_state_dict = {}
for k, v in encoder.state_dict().items():
    k = "conv_blocks." + k
    new_state_dict[k] = v
new_state_dict
# %%
for k, v in new_state_dict.items():
    if k in classifier.model.state_dict().keys():
        classifier.model.state_dict()[k] = v
        print(f"Set {k}")

# %%
# Save the classifier checkpoint
torch.save(
    classifier.model.state_dict(),
    "lightning_logs/checkpoints/convnet3d-large-classifier/icae-checkpoint-1.ckpt",
)

# %%
weights = torch.load(
    "lightning_logs/checkpoints/convnet3d-large-classifier/icae-checkpoint-1.ckpt"
)
weights
# %%
classifier = Classifier3D()

# %%
classifier.model.load_state_dict(weights)
# %%
classifier.model
# %%
classifier_freezed = Classifier3D(freeze_block=1)

# %%
