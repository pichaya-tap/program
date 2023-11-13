import numpy as np
import pandas as pd
import glob
import os
import sys
import getopt
import itertools
import time


def main(argv):

    # Initialize ENERGY and FILE variables
    ENERGY = None
    FILE = None

    # Get command line options and arguments
    opts, args = getopt.getopt(sys.argv[1:],"hi:e:f:",["ifile=","ofile="])

    # Process the command line options and arguments
    for opt, arg in opts:
        if opt == "-e":
            ENERGY = arg
        elif opt == "-f":
            FILE = arg
        else:
            print('Unknown Argument!')

    # Check if ENERGY and FILE have been provided
    if ENERGY is not None and FILE is not None:
        # Both ENERGY and FILE arguments are provided
        print(f'ENERGY: {ENERGY}')
        print(f'FILE: {FILE}')
    elif ENERGY is not None:
        # Only ENERGY argument is provided
        print(f'ENERGY: {ENERGY}')
        print('FILE argument is missing.')
    elif FILE is not None:
        # Only FILE argument is provided
        print('ENERGY argument is missing.')
    else:
        # Neither ENERGY nor FILE arguments are provided
        print('ENERGY and FILE arguments are missing.')


    #####################################Refer to "'notes_dicom_images.txt"########################################################
    DATASET = 'DATASET'
    POSITIONY = [0.1, 25, 50, 75]  # -50, -25,
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

    SOURCE_POSITION_XY = 0
    SOURCE_POSITION_Z = -200

    
    DENSITY_WATER = 1000 #kg/m^3
    WATERMASS_PER_VOXEL = DENSITY_WATER*DELTA_X*DELTA_X*DELTA_Z*(1e-9) #kg # need to convert mm^3 to m^3.

    elementarladung = 1.602176634e-19

    ##########################################################################################

    # Set the directory path where the files are located
    directory_path = '/home/tappay01/data/water/{}'.format(ENERGY)
    save_path = '/scratch/tappay01/watersimulation/{}'.format(DATASET)
    # Get a list of files matching the pattern "output_nt_Hits_t*.csv"
    print("Process file: ", FILE)
    file_pattern = os.path.join(directory_path, "output_nt_Hits_t{}.csv".format(FILE))
    inputfiles = glob.glob(file_pattern)

    # Initialize an empty DataFrame
    df = pd.DataFrame()
    
    for positiony, positionz in POSITION:
        dosemap = np.zeros(shape=(VOXELNUMBER_Z, VOXELNUMBER_Y, VOXELNUMBER_X))
        print(f"calculating for : y:{positiony} z:{positionz}")
        # create empty array to collect the sum of dosemaps, file name and path for saving
        filename_sum = '{}_{}_{}_{}.npy'.format(DATASET, ENERGY, positiony, positionz) 
        output_path = os.path.join(save_path, filename_sum)

        # check if the sum target file exists already, skip
        if os.path.exists(output_path):
            print(f"Warning: File '{output_path}' already exists. Skipping calculation.")
            continue

        # Construct the save_filename based on the file name and position values
        filename = '{}_t{}_{}_{}.npy'.format(ENERGY,FILE,positiony,positionz)
        file_path = os.path.join(save_path, filename)

        # Check if the file does not exist and df is empty
        if not os.path.exists(file_path) and df.empty:
            print("File does not exist. Read file before calculating....")
            # Read csv and perform calculations
            print('-- Loading File {} --'.format(file_pattern))           
            start_time = time.time()  # Record the start time         
            df = pd.read_csv(file_pattern, sep = ',', skiprows = 11, names = ['X', 'Y', 'Z', 'Edep', 'Process', 'particle', 'E'])
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time in seconds
            print(f"Time taken to read file: {elapsed_time} seconds")  
            data = df
            

        elif not os.path.exists(file_path):
            # Perform calculations based on the existing df
            data = df
            print("Calculating...")
        else:
            print(f"Warning: File '{file_path}' already exists. Skipping calculation.")
            continue

        start_time = time.time()  # Record the start time         
        data_edep = data['Edep'].values
        
        # source position_brain (-110, POSITIONY,POSITIONZ).New source position_water (0,0,-200) 
        # Add Delta to all edep positions. Unit in mm
        data['X'] = data['X'] + positionz
        data['Y'] = data['Y'] + positiony
        data['Z'] = data['Z'] + 90
                    
        # we want shape (283,283,160) -> (z,y,x)
        data_for_hist = data[['Z', 'Y', 'X']]                  
        
        dosemap, edges = np.histogramdd(data_for_hist.values, bins = (VOXELNUMBER_Z, VOXELNUMBER_Y, VOXELNUMBER_X), range = ((-200, 200),(-200, 200),(-200, 200)), weights = data_edep)
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time in seconds
        print(f"Time taken for histogram: {elapsed_time} seconds")
        #dosemap_shape = (283, 283, 160)
        print(dosemap.shape)

        # Umrechnen in Gray (Gy = J/kg und eV = e* J )
        dosemap = dosemap*(1e6)*elementarladung / WATERMASS_PER_VOXEL # from MeV
        # Save current dosemap
        np.save(file_path, dosemap)
        print('{} file added successfully'.format(filename))      


if __name__ == "__main__":
    main(sys.argv[1:])
