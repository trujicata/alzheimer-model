#%%
import start
import torch
import torch.nn as nn
#%%

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        
        # Define the same layers as in your Keras model
        self.conv1 = nn.Conv3d(1, 5, kernel_size=3, padding=1)  # Corrected input channels from 5 to 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(5)
        
        self.conv2 = nn.Conv3d(5, 5, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm3d(5)
        
        self.conv3 = nn.Conv3d(5, 5, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm3d(5)
        
        self.flatten = nn.Flatten()
        
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10800, 64)
        
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.maxpool3(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = nn.Softmax(dim=1)(x)
        return x

model = PyTorchModel()
# %%
random_input = torch.rand((1, 1, 100, 100, 120))
with torch.no_grad():
    output = model(random_input)
output.shape
# %%
weights_path = "data/resmodel_wb_cv9.best.hdf5"
#%%
import h5py

weights = h5py.File(weights_path, "r")
# %%
weights["model_weights"].keys()
# %%
from keras.models import load_model

keras_model = load_model(weights_path)
# %%

pytorch_state_dict = model.state_dict()
# %%
# First layer: convolutional
if keras_model.layers[0].get_weights()[0].transpose().shape == pytorch_state_dict["conv1.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["conv1.weight"] = torch.tensor(keras_model.layers[0].get_weights()[0].transpose())

if keras_model.layers[0].get_weights()[1].shape == pytorch_state_dict["conv1.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["conv1.bias"] = torch.tensor(keras_model.layers[0].get_weights()[1])

# %%
# First layer: batch normalization
weight, bias, running_mean, running_var = keras_model.layers[2].get_weights()

if weight.shape == pytorch_state_dict["bn1.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn1.weight"] = torch.tensor(weight)

if bias.shape == pytorch_state_dict["bn1.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn1.bias"] = torch.tensor(bias)

if running_mean.shape == pytorch_state_dict["bn1.running_mean"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn1.running_mean"] = torch.tensor(running_mean)

if running_var.shape == pytorch_state_dict["bn1.running_var"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn1.running_var"] = torch.tensor(running_var)
# %%
# Second layer: convolutional

if keras_model.layers[3].get_weights()[0].transpose().shape == pytorch_state_dict["conv2.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["conv2.weight"] = torch.tensor(keras_model.layers[3].get_weights()[0].transpose())

if keras_model.layers[3].get_weights()[1].shape == pytorch_state_dict["conv2.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["conv2.bias"] = torch.tensor(keras_model.layers[3].get_weights()[1])
# %%
# Second layer: batch normalization
weight, bias, running_mean, running_var = keras_model.layers[5].get_weights()

if weight.shape == pytorch_state_dict["bn2.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn2.weight"] = torch.tensor(weight)

if bias.shape == pytorch_state_dict["bn2.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn2.bias"] = torch.tensor(bias)

if running_mean.shape == pytorch_state_dict["bn2.running_mean"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn2.running_mean"] = torch.tensor(running_mean)

if running_var.shape == pytorch_state_dict["bn2.running_var"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn2.running_var"] = torch.tensor(running_var)
# %%
# Third layer: convolutional

if keras_model.layers[6].get_weights()[0].transpose().shape == pytorch_state_dict["conv3.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["conv3.weight"] = torch.tensor(keras_model.layers[6].get_weights()[0].transpose())

if keras_model.layers[6].get_weights()[1].shape == pytorch_state_dict["conv3.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["conv3.bias"] = torch.tensor(keras_model.layers[6].get_weights()[1])
# %%
# Third layer: batch normalization
weight, bias, running_mean, running_var = keras_model.layers[8].get_weights()

if weight.shape == pytorch_state_dict["bn3.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn3.weight"] = torch.tensor(weight)

if bias.shape == pytorch_state_dict["bn3.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn3.bias"] = torch.tensor(bias)

if running_mean.shape == pytorch_state_dict["bn3.running_mean"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn3.running_mean"] = torch.tensor(running_mean)

if running_var.shape == pytorch_state_dict["bn3.running_var"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["bn3.running_var"] = torch.tensor(running_var)
# %%
# First fully connected layer
weight, bias = keras_model.layers[11].get_weights()

if weight.transpose().shape == pytorch_state_dict["fc1.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["fc1.weight"] = torch.tensor(weight.transpose())

if bias.shape == pytorch_state_dict["fc1.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["fc1.bias"] = torch.tensor(bias)
# %%
# Second fully connected layer
weight, bias = keras_model.layers[13].get_weights()

if weight.transpose().shape == pytorch_state_dict["fc2.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["fc2.weight"] = torch.tensor(weight.transpose())

if bias.shape == pytorch_state_dict["fc2.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["fc2.bias"] = torch.tensor(bias)
# %%
# Third fully connected layer
weight, bias = keras_model.layers[15].get_weights()

if weight.transpose().shape == pytorch_state_dict["fc3.weight"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["fc3.weight"] = torch.tensor(weight.transpose())

if bias.shape == pytorch_state_dict["fc3.bias"].shape:
    print("Puting the weights in the correct order")
    pytorch_state_dict["fc3.bias"] = torch.tensor(bias)
# %%
model.load_state_dict(pytorch_state_dict)
# %%
# Save the model

torch.save(model.state_dict(), "data/resmodel_wb_cv9.best.pt")
# %%
