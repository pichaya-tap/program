'''
The code converts linear indices into corresponding three-dimensional coordinates
(x, y, z) within a voxel grid system.

caution: some filename with comma. Need to rename to get rid of comma after conversion.

'''

import pandas as pd
import numpy as np
import os
import time


# Shape of voxel
VOXELNUMBER_X = 256
VOXELNUMBER_Y = 256
VOXELNUMBER_Z = 126

# Location of the data
directory_path =  '/gpfs001/scratch/schwar14/simudosemapheavyions/DosemapInBrain/build_folders/Miriam/'
# Save path
save_path = '/home/tappay01/data/data1'

# Iterate through files in the folder
for filename in os.listdir(directory_path):
    if filename.endswith('.out') and filename.startswith('Data1_3000MeV'):
        # Start the timer
        start_time = time.time()
        # Construct the input and output file paths
        input_filename = os.path.join(directory_path, filename)
        
        # Get the file name without extension
        file_name_without_extension = os.path.splitext(filename)[0]

        # Output file name to change the extension to .npy
        output_filename = os.path.join(save_path, file_name_without_extension + '.npy')

        if not os.path.exists(output_filename):
            # if it doesn't exist already
            print('calculating ',file_name_without_extension)
            # Read the data from the input file     
            dicom_data = pd.read_table(input_filename, names=['Index', 'Dose'], sep='     ', engine='python')

            # Create an index array from 0 to VOXELNUMBER_X * VOXELNUMBER_Y * VOXELNUMBER_Z
            index_array = np.arange(VOXELNUMBER_X * VOXELNUMBER_Y * VOXELNUMBER_Z)

            # Filter the dicom_data DataFrame for matching indices
            filtered_data = dicom_data[dicom_data['Index'].isin(index_array)]
            
            # Create an empty dosemap array
            dosemap = np.zeros(shape=(VOXELNUMBER_X, VOXELNUMBER_Y, VOXELNUMBER_Z))

            # Calculate the x, y, z indices from the filtered data
            x_indices = (filtered_data['Index'] // VOXELNUMBER_Z) % VOXELNUMBER_Y
            y_indices = filtered_data['Index'] // (VOXELNUMBER_Z * VOXELNUMBER_Y)
            z_indices = filtered_data['Index'] % VOXELNUMBER_Z

            # Assign the dose values to the corresponding positions in the dosemap array
            dosemap[x_indices, y_indices, z_indices] = filtered_data['Dose'].values
            
            # Save the array to a file
            np.save(output_filename, dosemap)
            print(f"Saving dosemap in {output_filename}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time :{elapsed_time} s")

        else:
            print(f"File '{file_name_without_extension}.npy' already exists'. Skipped converting.")





