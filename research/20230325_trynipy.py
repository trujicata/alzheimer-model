#%%
import start
import nilearn.image as nl_image

#%%
img = nl_image.load_img(
    "data/PET - ADNI Nifti/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165149569_77_S165970_I330092.nii"
)

# %%
from nilearn import plotting

plotting.plot_img(
    "data/PET - ADNI Nifti/ADNI/137_S_1414/ADNI-2-AV-45_AV45/2012-09-04_14_34_53.0/I330092/ADNI_137_S_1414_PT_ADNI-2-AV-45_AV45_br_raw_20120904165149569_77_S165970_I330092.nii"
)

#%%
