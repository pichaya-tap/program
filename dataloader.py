import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import random
from torch.utils.data import Subset

# Write a custom dataset class (inherits from torch.utils.data.Dataset)
class CustomDataset(Dataset):

    # Initialize with a data_folder, conditional_folder, density_file path.
    def __init__(self, data_folder, conditional_folder, density_file):
        # Create class attributes
        # Get all data paths
        self.paths = list(Path(data_folder).glob("*.npy"))
        self.dosemap_water_folder = conditional_folder
        self.density = np.load(density_file).transpose((1, 2, 0))
        # calculate normalization parameters for both data and conditional input
        self.max_data = calculate_normalization_params(data_folder)
        self.max_dosemap_water = calculate_normalization_params(conditional_folder)

    # Overwrite the __len__() method 
    def __len__(self)-> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # Transform with normalization with maximum value, resize, turn to torch and unsqueeze
    def __transform__(self, data, max_value):
        normalized_data = data / max_value
        resized_data = resize(normalized_data)
        transformed_data = torch.from_numpy(resized_data).unsqueeze(dim=0)
        return transformed_data

    # Overwrite the __getitem__() method 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        "Returns one sample of data and its matching condition input."
        data_path = self.paths[idx]
        # Extract the condition parameters from the data_file name
        filename = os.path.basename(data_path) # Example "Data1_1500MeV_0Mm_-121.5Mm.npy"
        energy = filename.split('_')[1] # Example energy=1500MeV
        y,z = filename.split('_')[2][:-2],filename.split('_')[3][:-6] # Example y=0 and z=-121.5

        # Load data from the .npy file
        print("Data path:", data_path)
        data = np.load(data_path)

        # Find the corresponding conditional input file based on the condition
        dosemap_water_file = f'DATASET_{energy}_{y}_{z}.npy' #Example DATASET_1500MeV_0_-121.5.npy
        #print("dosemap_water_file:",dosemap_water_file)
        dosemap_water_path = os.path.join(self.dosemap_water_folder , dosemap_water_file)

        # Check if the water_dosemap file exists
        if not os.path.exists(dosemap_water_path): # If not, skip this data sample
            return None  
     
        # Load the conditional input from the .npy file
        dosemap_water =  np.load(dosemap_water_path)

        # Transform
        transformed_data = self.__transform__(data, self.max_data )
        transformed_dosemap_water = self.__transform__(dosemap_water, self.max_dosemap_water)
        transformed_density = self.__transform__(self.density, np.max(self.density))

        # Concatenate dosemap in water and material density as conditional_input
        conditional_input = torch.cat([transformed_density, transformed_dosemap_water], dim=0) # in the shape (2,256,256,128)

        return transformed_data, conditional_input



def resize(data):
    """Resize with either padding or truncating
    Compare current data shape with target size then perform resize.
    
    Args:
        data: A 3D numpy array data of any shape.
    Returns:
         A 3D numpy array data with shape of target size.
    """
    target_size = (256, 256, 128)
    current_size = data.shape

    if current_size == target_size:
        return data  # No need to resize

    if current_size[0] >= target_size[0] and \
       current_size[1] >= target_size[1] and \
       current_size[2] >= target_size[2]:
        # Truncate the data
        start_idx = (current_size[0] - target_size[0]) // 2
        end_idx = start_idx + target_size[0]
        truncated_data = data[start_idx:end_idx, :, :target_size[2]]
        return truncated_data

    # Padding the data
    padding = [(0, max(target_size[i] - current_size[i], 0)) for i in range(3)]
    padded_data = np.pad(data, padding, mode='constant')
    return padded_data


def calculate_normalization_params(data_folder):
    """Calculate normalization parameters
    Find maximum value among all data files with extension.npy in the directory.
    
    Args:
        data_folder: A directory where data is located.
    Returns:
         A maximum value of the data.
    """
    max_value = 0
    file_names = os.listdir(data_folder)
    for file_name in file_names:
        file_path = os.path.join(data_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".npy"):
            data = np.load(file_path)
            local_max = np.max(data)
            max_value = max(max_value, local_max)
    return max_value


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

    
        

