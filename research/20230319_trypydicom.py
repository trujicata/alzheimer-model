#%%
import start
from pydicom import dcmread
from matplotlib import pyplot as plt

#%%
path = "data/Datos CUDIM prueba FDG y PIB/FDG URY 1/FDG-URY-01_16903_13_REC OSEM_PT_1.2.840.113619.2.55.3.2831168258.572.1526330102.461.0158.dcm"
ds = dcmread(path)

# %%
# Transform to numpy
arr = ds.pixel_array
arr

#%%
# Plot image

plt.imshow(arr, cmap="gray")
plt.show()
# %%
# PatientID
ds.PatientID
# %%
