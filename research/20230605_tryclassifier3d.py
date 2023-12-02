# %%
import start  # noqa
import numpy as np
# from models.classifier3D.model import Classifier3D
from models.core.dataset import ADNIDataModule

# %%
datamodule = ADNIDataModule(
    data_path="data", num_workers=1
)

# %%
datamodule.setup("eval")

# %%
train_dataset = datamodule.train_dataset
# %%
sample = train_dataset[0]
img = sample["image"]

# Save img as a npy file
np.save("sample.npy", img.numpy())

# %%
model = Classifier3D(input_size=[181, 181, 215])
model.eval()
# %%
logits = model(img[:,:,:215].unsqueeze(0))
# %%
import time

start = time.time()
hola = np.load("sample.npy")
print(time.time() - start)
# %%
import torchvision.models as models
r3d_18 = models.video.r3d_18(pretrained=True)

# %%
r3d_18(img)
# %%
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=217,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)
# %%
model.encoder(img.permute(0,3,1,2))
# %%
