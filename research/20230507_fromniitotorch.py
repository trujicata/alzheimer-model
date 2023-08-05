#%%
import start
import nibabel as nib
import numpy as np
from PIL import Image

#%%
# Single image to numpy
img_1 = nib.load(
    "/home/catalina/workspace/alzheimer-model/data/PET/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165151377_159_S165970_I330092.nii"
)
#%%
img_np = np.array(img_1.dataobj)
# %%
import matplotlib.pyplot as plt

# Display the image using Matplotlib
for i in range(img_np.shape[2]):
    plt.imshow(img_np[:, :, i, 0], cmap="gray")
    plt.colorbar()
    plt.title("PET Image")
    plt.show()
# %%
# Data processing
# 1. Sacar las capas de los extremos porque pueden hacer ruido\
# 2. Bajar la profundidad
# 3. Hacer un promedio con todos los frames (siempre 3 primeros)
# 4. Normalizar max min
# 5. Cargar todas las imágenes y generar un dataset
# 6. Investigar qué tipo de transforms se le hacen a las imágenes de PET

#%%
# Modelado
# 1. Estado del arte-> Conv3D, X-Unet
# 2. Entrenar:
#   2.1 Algo sólo con las imágenes
#   2.2 Imágenes y metadata
