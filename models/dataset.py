import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

def peel_nii(img, num_layers):
    """Peel the first and last layers of the image"""
    peeled_image = img[:, :, num_layers:-num_layers, :]
    return peeled_image

def scale_array(arr):
    """Scale the array to 0-255"""
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = np.interp(arr, (min_val, max_val), (0, 255))
    scaled_arr = np.round(scaled_arr).astype(int)
    return scaled_arr

def scale_nii(img_np):
    """Scale all the nii file to 0-255"""
    for i in range(img_np.shape[2]):
        img_np[:, :, i, 0] = scale_array(img_np[:, :, i, 0])
    return img_np

def average_nii(nii_path):
    """Average the first three pet scans of a subject"""
    pet_scans = []
    for nii_file in os.listdir(nii_path)[:3]:
        pet_scan = nib.load(os.path.join(nii_path, nii_file))
        pet_scans.append(np.array(pet_scan.dataobj))
    avg_pet_scan = np.mean(np.array(pet_scans), axis=0)
    return avg_pet_scan
    
    
    

class PETDataset(Dataset):
    def __init__(self, pet_path) -> None:
        self.pet_path = pet_path
        csv_path = os.path.join(pet_path, "PET.csv")
        self.metadata = self._load_csv(csv_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        subject_id = self.metadata[idx, 0]
        label = self.metadata[idx, 1]

    def _load_csv(self, csv_file):
        try:
            data = np.genfromtxt(csv_file, delimiter=",", dtype=str)

            subject_id = data[1:, 0]
            group = data[1:, 2]

            metadata = np.hstack((subject_id, group))
            return metadata
        except FileNotFoundError:
            print("File not found. Please provide a valid CSV file.")

    def _process_nii(self,subject_files):
        #TODO: Process the nii files
