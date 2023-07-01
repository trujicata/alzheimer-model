#%%
# import start
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_laplace, convolve
from skimage import transform
import scipy.ndimage
import matplotlib.pyplot as plt

#%%
# Single image to numpy
img_1 = nib.load(
    "/Users/bruno/Documents/DataScience/ALZHEIMER/alzheimer-model/data/PET/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165149569_77_S165970_I330092.nii"
)
#%%
img_np = np.array(img_1.dataobj)
# %%


# Display the image using Matplotlib
def show_nii(img_np):
    for i in range(img_np.shape[2]):
        plt.imshow(img_np[:, :, i, 0], cmap="gray")
        plt.colorbar()
        plt.title("PET Image")
        plt.show()


# %%
# 1 . Sacar las capas de los extremos porque pueden hacer ruido
def peel_nii(img, num_layers):
    peeled_image = img[:, :, num_layers:-num_layers, :]
    return peeled_image


# %%
peeled = peel_nii(img_np, 3)
# %%
# show_nii(peeled)
# %%
# 2. Bajar la profundidad
def scale_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = np.interp(arr, (min_val, max_val), (0, 255))
    scaled_arr = np.round(scaled_arr).astype(int)
    return scaled_arr


#%%
def scale_nii(img_np):
    for i in range(img_np.shape[2]):
        img_np[:, :, i, 0] = scale_array(img_np[:, :, i, 0])
    return img_np


# %%
scaled_peeled = scale_nii(peeled)

# %%
# show_nii(scaled_peeled)
# %%
# 3. Hacer un promedio con todos los frames (siempre 3 primeros)
img_2 = nib.load(
    "/Users/bruno/Documents/DataScience/ALZHEIMER/alzheimer-model/data/PET/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165151377_159_S165970_I330092.nii"
)
img2_np = np.array(img_2.dataobj)
img_3 = nib.load(
    "/Users/bruno/Documents/DataScience/ALZHEIMER/alzheimer-model/data/PET/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165152374_36_S165970_I330092.nii"
)
img3_np = np.array(img_3.dataobj)

# %%
img_promedio = np.mean(np.array([img_np, img2_np, img3_np]), axis=0)
img_promedio.shape

# %%
# show_nii(img_promedio)
# %%
peeled_promedio = peel_nii(img_promedio, 3)
# show_nii(peeled_promedio)
# %%
scaled_peeled_promedio = scale_nii(peeled_promedio)
# show_nii(scaled_peeled_promedio)
# %%
#GAUSSIAN NOISE
mean_gaussian_noise = 0.0
stddev_gaussian_noise = 10
gaussian_noise = np.random.normal(mean_gaussian_noise, stddev_gaussian_noise, size=scaled_peeled_promedio.shape)
gaussiannoisy_scaled_peel_promedio= scaled_peeled_promedio + gaussian_noise
# show_nii(gaussiannoisy_scaled_peel_promedio)
show_nii(scaled_peeled_promedio[:,:,10:11,:])
show_nii(gaussiannoisy_scaled_peel_promedio[:,:,10:11,:])

# %%
#UNIFORM RANDOM NOISE
min_val_uniform = -30
max_val_uniform = 30
uniform_noise = np.random.uniform(min_val_uniform, max_val_uniform, size=scaled_peeled_promedio.shape)
uniformnoisy_scaled_peel_promedio= scaled_peeled_promedio + uniform_noise
# show_nii(uniformnoisy_scaled_peel_promedio)
show_nii(scaled_peeled_promedio[:,:,10:11,:])
show_nii(uniformnoisy_scaled_peel_promedio[:,:,10:11,:])

# %%
#GAUSSIAN BLUR
sigma_blur = 1.0
blurred_scaled_peel_promedio = gaussian_filter(scaled_peeled_promedio, sigma_blur)
# show_nii(blurred_scaled_peel_promedio)
show_nii(scaled_peeled_promedio[:,:,10:11,:])
show_nii(blurred_scaled_peel_promedio[:,:,10:11,:])


# %%
#SHARPEN IMAGE
sigma_sharpen = 1.0
smoothed_image = gaussian_filter(scaled_peeled_promedio, sigma_sharpen)
laplacian_image = gaussian_laplace(smoothed_image, sigma_sharpen)
sharpening_factor = 2
sharpen_scaled_peel_promedio = scaled_peeled_promedio + (sharpening_factor * laplacian_image)
# show_nii(sharpen_scaled_peel_promedio)
show_nii(scaled_peeled_promedio[:,:,10:11,:])
show_nii(sharpen_scaled_peel_promedio[:,:,10:11,:])

# %%
#EMBOSS IMAGE
# Reshape the data if necessary
emboss_data = np.squeeze(scaled_peeled_promedio)  # Remove singleton dimensions if present
emboss_data = np.transpose(emboss_data, (2, 0, 1))  # Transpose to match (z, x, y) order

# Define the emboss kernel
kernel = np.array([[0, -3, -3],
                   [3,  0, -3],
                   [3,  3,  0]])

# Emboss the image
embossed_scaled_peeled_promedio = np.zeros_like(emboss_data)
for i in range(emboss_data.shape[0]):
    embossed_scaled_peeled_promedio[i] = convolve(emboss_data[i], kernel)

# show_nii(sharpen_scaled_peel_promedio)
show_nii(scaled_peeled_promedio[:,:,10:11,:])
show_nii(sharpen_scaled_peel_promedio[:,:,10:11])


# %%
#ELASTIC DEFORMATION

def elastic_deformation(image, alpha, sigma):
    # Generate random displacement field
    shape = image.shape
    displacement_field = np.random.randn(*shape) * alpha
    
    # Smooth the displacement field
    smoothed_field = scipy.ndimage.gaussian_filter(displacement_field, sigma=sigma, mode='reflect')
    
    # Generate grid coordinates
    grid = np.meshgrid(*[np.arange(dim) for dim in shape], indexing='ij')
    indices = [coord + smoothed_field.astype(int) for coord in grid]
    
    # Perform elastic deformation
    deformed_image = scipy.ndimage.map_coordinates(image, indices, mode='reflect')
    deformed_image = deformed_image.reshape(shape)
    
    return deformed_image

alpha = 2000  # Deformation magnitude
sigma = 20  # Smoothing parameter

deformed_image = elastic_deformation(scaled_peeled_promedio, alpha, sigma)

show_nii(scaled_peeled_promedio[:,:,10:11,:])
show_nii(deformed_image[:,:,10:11,:])


# %%
#RANDOM DISPLACEMENT 

def random_displacement(image, max_displacement):
    # Generate random displacements for each pixel
    displacements = np.random.randint(-max_displacement, max_displacement + 1, size=image.shape)
    
    # Apply displacements to the image
    displaced_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                displaced_image[i, j, k] = image[i, j, k] + displacements[i, j, k]
    
    return displaced_image


max_displacement = 30  # Maximum displacement for each pixel

displaced_image = random_displacement(scaled_peeled_promedio, max_displacement)


show_nii(scaled_peeled_promedio[:,:,10:11,:])
show_nii(displaced_image[:,:,10:11,:])

