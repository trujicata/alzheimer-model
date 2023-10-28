# %%
import h5py
import numpy as np
import start
import torch

from models.classifier3D.model import ConvNet

#%%
train_path = "data/train.hdf5"
test_path = "data/test.hdf5"
holdout_path = "data/holdout.hdf5"
# %%
train_h5 = h5py.File(train_path, "r")
train_data = train_h5["X_nii"]
label_data = train_h5["y"]
one = train_data[90]

# %%
model = ConvNet()
weights = torch.load("data/convnet.pt")
# Load the weights from old model to new model minus all the classifier.2 layer

for k,v in weights.items():
    if "classifier.2" not in k:
        print(k)
        model.state_dict()[k].copy_(v)
    else:
        print("Skipping", k)


#%%
model.load_state_dict(torch.load("data/convnet_new_name.pt"))
# %%
one_reshaped = one[40:160, 58:158, 40:140]
one_reshaped.shape
#%%
import matplotlib.pyplot as plt

# Display the image using Matplotlib
def show_nii(img_np):
    random_indexes = np.random.randint(30, img_np.shape[2], 15)
    for i in random_indexes:
        plt.imshow(img_np[:, i, :], cmap="gray")
        plt.colorbar()
        plt.title("PET Image")
        plt.show()

show_nii(one_reshaped)
# %%
model_input = torch.from_numpy(one_reshaped).unsqueeze(0).unsqueeze(0).float()
with torch.no_grad():
    out = model(model_input)
    print(out)
# %%
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
baseline = torch.zeros_like(model_input)
attributions, delta = ig.attribute(model_input, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)
# %%
attributions_numpy = attributions.squeeze(0).squeeze(0).numpy()
attributions_numpy.shape
# %%
show_nii(attributions_numpy)
# %%
