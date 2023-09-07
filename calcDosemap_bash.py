import numpy as np
import pandas as pd
import glob
import os
import sys
import getopt
import itertools


def main(argv):

    opts, args = getopt.getopt(argv,"hi:e:",["ifile=","ofile="])
    for opt, arg in opts:
        if opt == "-e":
            ENERGY = arg
        else:
            print('Unknown Argument!')

    #####################################Refer to "'notes_dicom_images.txt"########################################################
    DATASET = 'DATASET'
    POSITIONY = [-50, -25, 0, 25, 50, 75]  
    POSITIONZ = [-146.5, -121.5, -96.5, -71.5, -46.5] 
    POSITION = itertools.product(POSITIONY, POSITIONZ)
    #XY_MIN = -181
    #XY_MAX = 181

    #Z_MIN = -229
    #Z_MAX = 86
    PHANTOM_SIZE = 400
    VOXELNUMBER_XY = int(PHANTOM_SIZE/DELTA_X) 
    VOXELNUMBER_Z = int(PHANTOM_SIZE/DELTA_Z)
    
    DELTA_X = 0.707031*2
    DELTA_Z = 2.5
    SOURCE_POSITION_XY = 0
    SOURCE_POSITION_Z = -200

    DENSITY_WATER = 1000 #kg/m^3
    WATERMASS_PER_VOXEL = DENSITY_WATER*DELTA_X*DELTA_X*DELTA_Z #kg

    elementarladung = 1.602176634e-19

    ##########################################################################################
    
    # Set the directory path where the files are located
    directory_path = '/gpfs001/scratch/schwar14/simudosemapheavyions/Watersimulation/Hadr01/build/{}'.format(ENERGY)
    save_path = '/scratch/tappay01/watersimulation/{}'.format(DATASET)
    # Get a list of files matching the pattern "output_nt_Hits_t*.csv"
    file_pattern = os.path.join(directory_path, "output_nt_Hits_t*.csv")
    inputfiles = glob.glob(file_pattern)
    
 
    # Perform the calculation for each pair 
    for positiony, positionz in POSITION:
        # create empty array to collect the sum of dosemaps, file name and path for saving
        dosemap_sum = np.zeros(shape=(VOXELNUMBER_XY, VOXELNUMBER_XY, VOXELNUMBER_Z))
        filename_sum = '{}_{}_{}_{}.npy'.format(DATASET, ENERGY, positiony, positionz) 
        output_path = os.path.join(save_path, filename_sum)

        # check if the target file exists already, skip
        if os.path.exists(output_path):
            print(f"Warning: File '{output_path}' already exists. Skipping calculation.")
            continue

        for file in inputfiles:
            print('-- Loading File {} --'.format(file))
            data = pd.read_csv(file, sep = ',', skiprows = 11, names = ['X', 'Y', 'Z', 'Edep', 'Process', 'particle', 'E'])
            data_edep = data['Edep'].values
            data = data.drop([ 'Edep', 'Process', 'particle', 'E'], axis = 1)
            print(data.head())
            # data['X'], data['Z'] = data['Z'].values, data['X'].values
            # Add Delta to all edep positions
            # source position_brain (-110, POSITIONY,POSITIONZ).New source position_water (0,0,-200) with beamdirection (0,0,1).
            # Delta = position_brain – position_water = (POSITIONZ, POSITIONY, 90) 
            # Add Delta to all edep positions. Unit in mm
            data['X'] = data['X'] + positionz  
            data['Y'] = data['Y'] + positiony
            data['Z'] = data['Z'] + 90                    
            
            dosemap, edges = np.histogramdd(data.values, bins = (VOXELNUMBER_XY, VOXELNUMBER_XY, VOXELNUMBER_Z), range = ((-200, 200),(-200, 200),(-200, 200)), weights = data_edep)
            
    #########################################################################################

            # Umrechnen in Gray (Gy = J/kg und eV = e* J )
            dosemap = dosemap*(1e-6)*elementarladung / WATERMASS_PER_VOXEL
            # Save current dosemap
            # Construct the save_filename based on the file name and position values
            save_filename = '{}_{}_{}_{}.npy'.format(ENERGY,file.split(".")[0].split("_")[-1],positiony,positionz)
            file_path = os.path.join(save_path, save_filename)
            np.save(file_path, dosemap)
            print('{} file saved successfully'.format(save_filename))

            # Add current dosemap to the overall dosemap
            dosemap_sum += dosemap
        
        # Save the overall dose map        
        np.save(output_path, dosemap_sum)
        print('{} file saved successfully'.format(filename_sum))

if __name__ == "__main__":
    main(sys.argv[1:])
