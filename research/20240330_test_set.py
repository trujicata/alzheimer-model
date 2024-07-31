# %%
import start
import torch

import matplotlib.pyplot as plt

from models.resnet.model import Classifier3D
from models.core.dataset import ADNIDataModule
from models.eval.dataset import ClassificationDataModule as EvalADNIDataModule

from captum.attr import IntegratedGradients

# %%
model = Classifier3D()

checkpoint_path = "lightning_logs/checkpoints/resnet/resnet-18/resnet-18-epoch=66-val_loss=0.54-val_f1=0.78.ckpt"

weights = torch.load(checkpoint_path)["state_dict"]

for k, v in model.state_dict().items():
    if k in weights.keys():
        model.state_dict()[k] = weights[k]
    else:
        print(f"Key {k} not found in checkpoint")

# %%
adni_datamodule = ADNIDataModule(
    data_path="data/original", batch_size=32, num_workers=4
)
adni_datamodule.setup("test")
# %%
eval_datamodule = EvalADNIDataModule(
    data_path="data/test-dataset",
    post_or_pre="post",
    model_name="ResNet",
    batch_size=32,
    num_workers=4,
)
eval_datamodule.setup("test")
# %%
train_image = adni_datamodule.train_dataset[0]["image"]


# Display the image using Matplotlib
def show_nii(img_np):
    for i in range(img_np.shape[2]):
        plt.imshow(img_np[:, i, :], cmap="gray")
        plt.colorbar()
        plt.title("PET Image")
        plt.show()


show_nii(train_image[0, :, :, :].detach().numpy())

# %%
input = train_image.unsqueeze(0)
baseline = torch.zeros_like(input)
input.shape

# %%
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(
    input, baseline, target=0, return_convergence_delta=True
)
print("IG Attributions:", attributions)
print("Convergence Delta:", delta)

# %%
for i in range(attributions.shape[2]):
    plt.imshow(attributions[0, 0, i, :, :].detach().numpy(), cmap="bwr")
    plt.colorbar()
    plt.title(f"Integrated Gradients: {i}")
    plt.show()


# %%
test_image = eval_datamodule.test_dataset[5]["image"]
tst_input = test_image.unsqueeze(0)
baseline = torch.zeros_like(tst_input)
print(tst_input.shape)

show_nii(test_image[0, :, :, :].detach().numpy())

# %%
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(
    tst_input, baseline, target=0, return_convergence_delta=True
)
print("IG Attributions:", attributions)
print("Convergence Delta:", delta)

# %%
for i in range(attributions.shape[2]):
    plt.imshow(attributions[0, 0, i, :, :].detach().numpy(), cmap="bwr")
    plt.colorbar()
    plt.title(f"Integrated Gradients: {i}")
    plt.show()

# %%
