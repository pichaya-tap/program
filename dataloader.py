import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import random
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
        return float(parts[2][:-2])
    if param == 'z':
        return float(parts[3][:-6])
    if param == 'energy':
        return parts[1]
    else:
        return None

def find_corresponding_file_in_folder(energy, folder):
    """Returns the path of the corresponding file based on the energy value."""
    target_filename = f"DATASET_{energy}.npy"
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

        # Extract samples first without normalization
        for data_file in data_files:
            data = np.load(data_file)
            filename = os.path.basename(data_file) # Example "Data1_1500MeV_0Mm_-121.5Mm.npy"
            y = extract_from_filename(filename, 'y')
            z = extract_from_filename(filename, 'z')  # Example y=0 and z=-121.5
            y_voxel = to_voxel(y,1)
            z_voxel = to_voxel(z,0)
            sample = extract_centered_subsample(data, z_voxel, y_voxel, x_range)
            self.data_samples.append(sample)  
            self.data_names.append(filename)

            density_file = filename.split('_')[0]+'.npy' # Example Data1, Data2
            density = np.load(os.path.join(density_folder, density_file))
            density_sample = extract_centered_subsample(density, z_voxel, y_voxel, x_range)
            self.density_samples.append(density_sample)


            energy = extract_from_filename(filename, 'energy')
            watersim_path = find_corresponding_file_in_folder(energy, water_folder)
            if not os.path.exists(watersim_path):                
                print(f"Water simulation file not found for {energy}")
                watersim_path = find_corresponding_file_in_folder('2750MeV', water_folder) #as default
            watersim = np.load(watersim_path)/10 #ratio of primary particles or histories (Water simulation n=10^8 and Phantom simulation n=10^7)
            water_sample= extract_centered_subsample(watersim, 80, 141, [60,188])
            self.water_samples.append(water_sample)
            
        # Thresholding
        dose_threshold= 0.1# Set a threshold for values consider close to 0
        density_threshold = 1.5 # Set a threshold to limit outliers
        self.data_samples = [np.where(np.abs(sample) < dose_threshold, 0, sample) for sample in self.data_samples]
        self.water_samples = [np.where(np.abs(sample) < dose_threshold, 0, sample) for sample in self.water_samples]
        self.density_samples = [np.where(sample > density_threshold, density_threshold, sample) 
                        for sample in self.density_samples]
        # Normalize the data_samples
        all_data = np.array(self.data_samples)
        if normalization == 'zscore':
            mean = np.mean(all_data)
            std = np.std(all_data)
            self.data_samples = [(sample - mean) / std for sample in self.data_samples]
            
        elif normalization == 'minmax':
            data_min = np.min(all_data)
            data_range = np.max(all_data) - data_min
            self.data_samples = [(sample - data_min) / data_range for sample in self.data_samples]

        # Similarly, normalize the density_samples 
        all_density = np.array(self.density_samples)
        if normalization == 'zscore':
            mean = np.mean(all_density)
            std = np.std(all_density)
            self.density_samples = [(sample - mean) / std for sample in self.density_samples]
       
        elif normalization == 'minmax':
            density_min = np.min(all_density)
            density_range = np.max(all_density) - density_min
            self.density_samples = [(sample - density_min) / density_range for sample in self.density_samples]
  

        # Similarly, normalize the water_samples 
        all_water = np.array(self.water_samples)
        if normalization == 'zscore':
            mean = np.mean(all_water)
            std = np.std(all_water)
            self.water_samples = [(sample - mean) / std for sample in self.water_samples]

        elif normalization == 'minmax':
            water_min = np.min(all_water)
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
        
        return data_tensor,condition, water_tensor, density_tensor, data_name




def resize(data,target_size):
    """Resize with either padding or truncating
    Compare current data shape with target size then perform resize.
    
    Args:
        data: A 3D numpy array data of any shape.
    Returns:
         A 3D numpy array data with shape of target size.
    """

    current_size = data.shape

    if current_size == target_size:
        return data  # No need to resize

    if current_size[0] >= target_size[0] and \
       current_size[1] >= target_size[1] and \
       current_size[2] >= target_size[2]:
        # Truncate the data
        start_idx = (current_size[0] - target_size[0]) // 2
        end_idx = start_idx + target_size[0]
        start_idx_1 = (current_size[1] - target_size[1]) // 2
        end_idx_1 = start_idx_1 + target_size[1]
        start_idx_2 = (current_size[2] - target_size[2]) // 2
        end_idx_2 = start_idx_2 + target_size[2]
        truncated_data = data[start_idx:end_idx, start_idx_1:end_idx_1, start_idx_2:end_idx_2]
        return truncated_data

    # Padding the data
    padding = [(0, max(target_size[i] - current_size[i], 0)) for i in range(3)]
    padded_data = np.pad(data, padding, mode='constant')

    return padded_data





def split(custom_dataset,
          train_ratio: float = 0.6, 
          val_ratio: float = 0.2,
          test_ratio: float = 0.2):
    """Split custom data into training, validation, testing subset"""

    #CustomDataset object named 'custom_dataset'
    data_size = len(custom_dataset)
    indices = list(range(data_size))
    random.shuffle(indices)

    # Calculate the sizes for each split
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create Subset objects for each split
    train_subset = Subset(custom_dataset, train_indices)
    val_subset = Subset(custom_dataset, val_indices)
    test_subset = Subset(custom_dataset, test_indices)

    return(train_subset, val_subset, test_subset)

    
        

