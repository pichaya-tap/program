import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple

from torch.utils.data import Subset
import glob

def to_voxel(position, axis):
    '''find voxel number as integer from position in mm. 
    Axis number representation 
    0 = z, 1 = y, 2 = x (beam)
    '''
    DELTA_Z = 2.5
    DELTA_X = 0.707031*2
    DELTA_Y = 0.707031*2
    if axis ==0:
        voxel = (position + 229) /DELTA_Z
    if axis ==1:
        voxel = (position + 181) /DELTA_Y
    if axis ==2:
        voxel = (position + 181) /DELTA_X
    return int(voxel) 


def extract_centered_subsample(data, z, y, x):
    half_z = 8  # half of 16
    half_y = 8  # half of 16
    sample = data[z-half_z:z+half_z, y-half_y:y+half_y, x[0]:x[1]]
    return sample

def extract_from_filename(filename, param):
    """Extracts energy value from the given filename."""
    parts = filename.split('_')
    if param == 'y':
        return int(parts[2][:-2])
    if param == 'z':
        return float(parts[3][:-6])
    if param == 'energy':
        return parts[1]
    else:
        return None

def find_watersim_in_folder(energy, folder):
    """Returns the path of the corresponding water simulation file based on the energy value."""
    target_filename = f"DATASET_{energy}.npy"
    return os.path.join(folder, target_filename)

def find_density_in_folder(dataset,y,z, folder):
    """Returns the path of the corresponding density file based on the dataset and source position value."""
    target_filename = f"{dataset}_{y}Mm_{z}Mm.npy"
    return os.path.join(folder, target_filename)



# Write a custom dataset class (inherits from torch.utils.data.Dataset)

class CustomDataset(Dataset):
    def __init__(self, data_folder, density_folder,water_folder, normalization='minmax'):
        x_range = [60, 188]
        
        self.data_samples = []
        self.density_samples = []

        self.data_names = []
        self.water_samples = []

        data_files = glob.glob(os.path.join(data_folder, "*.npy"))           
        print(data_folder)
        # Extract samples first without normalization
        for data_file in data_files:
            data = np.load(data_file)

            # Check if the data array is empty or has any zero dimension
            if data.size == 0 or 0 in data.shape:
                print(f"Invalid data file (empty or zero dimension): {data_file}")
                continue  # Skip the rest of the loop and go to the next file

            # Check if all values in the array are zero
            if np.all(data == 0):
                print(f"All values are zero in file: {data_file}")
                continue  # Skip the rest of the loop and go to the next file

            # Water simulation
            filename = os.path.basename(data_file) # Example "Data1_1500MeV_0Mm_-121.5Mm.npy"
            energy = extract_from_filename(filename, 'energy')
            watersim_path = find_watersim_in_folder(energy, water_folder)
            if not os.path.exists(watersim_path):                
                print(f"Water simulation file not found for {energy}")
                #watersim_path = find_watersim_in_folder('3000MeV', water_folder) #as default
                continue
            watersim = np.load(watersim_path) #ratio of primary particles or histories (Water simulation n=10^8 and Phantom simulation n=10^7)
            water_sample= extract_centered_subsample(watersim, 80, 141, x_range) # watersim shape zyx 160*283*283

            # Densities
            dataset = filename.split('_')[0] # Example Data1, Data2
            y = extract_from_filename(filename, 'y')
            z = extract_from_filename(filename, 'z')  # Example y=0 and z=-121.5
            density = find_density_in_folder(dataset,y,z, density_folder)
            density_sample = np.load(density)
            
            self.data_samples.append(data)
            self.water_samples.append(water_sample)  
            self.density_samples.append(density_sample)
            self.data_names.append(filename)

           
        # Thresholding
        dose_threshold= 0.1# Set a threshold for values consider close to 0
        density_threshold = 1.5 # Set a threshold to limit outliers
        self.data_samples = [np.where(np.abs(sample) < dose_threshold, 0, sample) for sample in self.data_samples]
        self.water_samples = [np.where(np.abs(sample) < dose_threshold, 0, sample) for sample in self.water_samples]
        self.density_samples = [np.where(sample > density_threshold, density_threshold, sample) 
                        for sample in self.density_samples]
        
        # Normalize 
        all_data = np.array(self.data_samples)
        print(all_data.shape)
        all_density = np.array(self.density_samples)
        all_water = np.array(self.water_samples)
        if normalization == 'minmax':
            data_min = 0
            data_range = np.max(all_data) - data_min
            self.data_samples = [(sample - data_min) / data_range for sample in self.data_samples]
  
            density_min = np.min(all_density)
            density_range = np.max(all_density) - density_min
            self.density_samples = [(sample - density_min) / density_range for sample in self.density_samples]
            
            water_min = 0
            water_range = np.max(all_water) - water_min
            self.water_samples = [(sample - water_min) / water_range for sample in self.water_samples]

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data_samples[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        density_tensor = torch.tensor(self.density_samples[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        data_name = self.data_names[idx]
        water_tensor = torch.tensor(self.water_samples[idx], dtype=torch.float32).unsqueeze(0) 
        # Concatenate along the channel dimension
        condition = torch.cat([water_tensor, density_tensor], dim=0)
        
        return data_tensor, condition, water_tensor, density_tensor, data_name




