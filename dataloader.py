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
        self.density = np.load(density_file).transpose((2, 1, 0))
        # calculate normalization parameters for both data and conditional input
        self.min_data, self.max_data = calculate_normalization_params(data_folder)
        self.min_dosemap_water, self.max_dosemap_water = calculate_normalization_params(conditional_folder)
        self.transformed_density = self.__transform__(self.density, np.min(self.density), np.max(self.density))
    # Overwrite the __len__() method 
    def __len__(self)-> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # Transform with normalization with maximum value, resize, turn to torch and unsqueeze
    def __transform__(self, data, min_value_global, max_value_global):
        #data = data.astype(np.float16) #Reduce Data Precision
        data = (data - min_value_global) / (max_value_global - min_value_global)  #In-place to save memory 
        data = resize(data,(256, 256, 128))[:,60:188,:] # get 256x128x128

        data = np.resize(data, (128,64,64))
        return torch.from_numpy(data).unsqueeze(dim=0)
    

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
        print("dosemap_water_file:",dosemap_water_file)
        dosemap_water_path = os.path.join(self.dosemap_water_folder , dosemap_water_file)

        # Check if the water_dosemap file exists
        if not os.path.exists(dosemap_water_path): # If not, skip this data sample
            dummy_data = torch.ones((1,128,64,64))
            dummy_conditional_input = torch.ones((2,128,64,64))
            # Avoid return None
            return dummy_data, dummy_conditional_input
     
        # Load the conditional input from the .npy file
        dosemap_water =  np.load(dosemap_water_path) 
  
        # Transform
        transformed_data = self.__transform__(data,self.min_data, self.max_data )
        transformed_dosemap_water = self.__transform__(dosemap_water,self.min_dosemap_water, self.max_dosemap_water)


        # Concatenate dosemap in water and material density as conditional_input
        conditional_input = torch.cat([self.transformed_density, transformed_dosemap_water], dim=0) # in the shape (2,256,256,128)
        return transformed_data , conditional_input



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



def calculate_normalization_params(data_folder):
    """Calculate normalization parameters
    Find maximum and minimum value among all data files with extension.npy in the directory.
    
    Args:
        data_folder: A directory where data is located.
    Returns:
         A minimum and maximum value of the data.
    """
    # Calculate global min and max values across the dataset
    print('calculate_normalization_params')
    min_value_global = np.inf
    max_value_global = -np.inf
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        # Check if the file is a valid numpy file (modify as needed)
        if file_name.endswith('.npy'):
            data = np.load(file_path)
            print(file_path)
            min_value_global = min(min_value_global, np.min(data))
            max_value_global = max(max_value_global, np.max(data))
            '''
            try:
                # Load the data from the file
                data = np.load(file_path)
                print(file_path)
                min_value_global = min(min_value_global, np.min(data))
                max_value_global = max(max_value_global, np.max(data))
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
            '''
    print('return min_value_global, max_value_global', min_value_global, max_value_global)
    return min_value_global, max_value_global


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

    
        

