'''
The code converts linear indices into corresponding three-dimensional coordinates
within a voxel grid system.
Dosemap shape matches density shape (z,y,x)	 for Dataset7 (136, 256, 256)
Beam axis is in last dimension


caution: some filename with comma. Need to rename to get rid of comma after conversion.

'''

import pandas as pd
import numpy as np
import os
import time


# Shape of voxel
VOXELNUMBER_X = 256
VOXELNUMBER_Y = 256
VOXELNUMBER_Z = 136

# Location of the data
directory_path =  '/gpfs001/scratch/schwar14/simudosemapheavyions/DosemapInBrain/build_folders/Hans/'
print('conversion files from ',directory_path)
# Save path
save_path = '/scratch/tappay01/data/data7'

# Iterate through files in the folder
for filename in os.listdir(directory_path):
    if filename.endswith('.out') and filename.startswith('Data7_2500MeV_75Mm_-96.5Mm'):
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
            dosemap = np.zeros(shape=(VOXELNUMBER_Z, VOXELNUMBER_Y, VOXELNUMBER_X))

            # Calculate the x, y, z indices from the filtered data
            y_indices = filtered_data['Index'] // (VOXELNUMBER_Z * VOXELNUMBER_Y)
            x_indices = (filtered_data['Index'] // VOXELNUMBER_Z) % VOXELNUMBER_Y            
            z_indices = filtered_data['Index'] % VOXELNUMBER_Z

            # Assign the dose values to the corresponding positions in the dosemap array
            dosemap[z_indices, y_indices, x_indices] = filtered_data['Dose'].values
            
            # Save the array to a file
            np.save(output_filename, dosemap)
            print(f"Saving dosemap in {output_filename}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time :{elapsed_time} s")

        else:
            print(f"File '{file_name_without_extension}.npy' already exists'. Skipped converting.")





