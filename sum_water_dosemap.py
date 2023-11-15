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

    PHANTOM_SIZE = 400
    DELTA_X = 2.5 #mm
    DELTA_Y = 0.707031*2
    DELTA_Z = 0.707031*2
    
    VOXELNUMBER_X = int(PHANTOM_SIZE/DELTA_X)
    #+1 weil ich in der Simu aufgerundet habe und int abrundet...
    VOXELNUMBER_Y = int(PHANTOM_SIZE/DELTA_Y)+1
    VOXELNUMBER_Z = int(PHANTOM_SIZE/DELTA_Z)+1
    directory_path = '/scratch/tappay01/watersimulation/DATASET/'
    # Get a list of files matching the pattern "output_nt_Hits_t*.csv"
    file_pattern = os.path.join(directory_path, "4250*.npy")
    inputfiles = glob.glob(file_pattern)
    dosemap_sum = np.zeros(shape=(VOXELNUMBER_X, VOXELNUMBER_Y, VOXELNUMBER_Z))
    # create empty array to collect the sum of dosemaps, file name and path for saving
    filename_sum = 'DATASET_{}.npy'.format(ENERGY)
    output_path = os.path.join(directory_path, filename_sum)

    for file in inputfiles:
        print(f"Sum file {file} ")
        dosemap_sum += np.load(file)

    # Save the overall dose map        
    np.save(output_path, dosemap_sum)
    print('{} file saved successfully'.format(filename_sum))   


if __name__ == "__main__":
    main(sys.argv[1:])
