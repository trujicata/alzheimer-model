# %%
import start
import torch
import torch.nn as nn
import numpy as np

from models.classifier3D.model import Classifier3D
from models.core.dataset import class_trad2


# %%
model = Classifier3D()

# %%
random_image = torch.rand((1, 1, 100, 100, 120))

with torch.no_grad():
    output = model(random_image)
output
# %%
output = torch.stack([output, output, output])
output
# %%
class_predictions = [class_trad2(x) for x in output]
preds = torch.zeros(output.shape[0], 3)
preds[torch.arange(output.shape[0]), class_predictions] = 1

# %%
