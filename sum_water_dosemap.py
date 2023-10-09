import numpy as np
import pandas as pd
import glob
import os
import sys
import getopt
import itertools
import time


def main(argv):

    opts, args = getopt.getopt(argv,"hi:e:",["ifile=","ofile="])
    for opt, arg in opts:
        if opt == "-e":
            ENERGY = arg
        else:
            print('Unknown Argument!')

    #####################################Refer to "'notes_dicom_images.txt"########################################################
    DATASET = 'DATASET'
    POSITIONY = [-50, -25, 0.1, 25, 50, 75]  
    POSITIONZ = [-146.5, -121.5, -96.5, -71.5, -46.5] 
    POSITION = itertools.product(POSITIONY, POSITIONZ)

    
    PHANTOM_SIZE = 400
    DELTA_X = 2.5 #mm
    DELTA_Y = 0.707031*2
    DELTA_Z = 0.707031*2
    
    VOXELNUMBER_X = int(PHANTOM_SIZE/DELTA_X)
    #+1 weil ich in der Simu aufgerundet habe und int abrundet...
    VOXELNUMBER_Y = int(PHANTOM_SIZE/DELTA_Y)+1
    VOXELNUMBER_Z = int(PHANTOM_SIZE/DELTA_Z)+1
    
    for positiony, positionz in POSITION:
        start_time = time.time()  # Record the start time 
        # create empty array to collect the sum of dosemaps, file name and path for saving
        dosemap_sum = np.zeros(shape=(VOXELNUMBER_Z, VOXELNUMBER_Y, VOXELNUMBER_X))
        filename_sum = '{}_{}_{}_{}.npy'.format(DATASET, ENERGY, positiony, positionz) 
        save_path = '/scratch/tappay01/watersimulation/{}'.format(DATASET)
        output_path = os.path.join(save_path, filename_sum)
        
        # check if the target file exists already, skip
        if os.path.exists(output_path):
            print(f"Warning: File '{output_path}' already exists. Skipping calculation.")
            continue  

        for i in range(25): 
            filename = '{}_t{}_{}_{}.npy'.format(ENERGY,i,positiony,positionz)
            file_path = os.path.join(save_path, filename)
            # Add current dosemap to the overall dosemap
            dosemap_sum += np.load(file_path)
            # Save the overall dose map   
             
        np.save(output_path, dosemap_sum)
        print('{} file saved successfully'.format(filename_sum))
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time in seconds
        print(f"Time taken for sum up files: {elapsed_time} seconds")


if __name__ == "__main__":
    main(sys.argv[1:])
