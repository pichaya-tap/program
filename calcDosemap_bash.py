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
    directory_path = '/gpfs001/scratch/schwar14/simudosemapheavyions/Watersimulation/Hadr01/Dataset_Miriam/{}'.format(ENERGY)
    save_path = '/scratch/tappay01/watersimulation/{}'.format(DATASET)
    # Get a list of files matching the pattern "output_nt_Hits_t*.csv"
    file_pattern = os.path.join(directory_path, "output_nt_Hits_t*.csv")
    inputfiles = glob.glob(file_pattern)
    
 
    # Perform the calculation for each pair 
    for positiony, positionz in POSITION:
        # create empty array to collect the sum of dosemaps, file name and path for saving
        dosemap_sum = np.zeros(shape=(VOXELNUMBER_Z, VOXELNUMBER_Y, VOXELNUMBER_X))
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
            

            # source position_brain (-110, POSITIONY,POSITIONZ).New source position_water (0,0,-200) 

            # Add Delta to all edep positions. Unit in mm
            data['X'] = data['X'] + positionz
            data['Y'] = data['Y'] + positiony
            data['Z'] = data['Z'] + 90
                        
            # we want shape (283,283,160) -> (z,y,x)
            data_for_hist = data[['Z', 'Y', 'X']]                  
            
            dosemap, edges = np.histogramdd(data_for_hist.values, bins = (VOXELNUMBER_Z, VOXELNUMBER_Y, VOXELNUMBER_X), range = ((-200, 200),(-200, 200),(-200, 200)), weights = data_edep)
            #dosemap_shape = (283, 283, 160)
            # edges_shape: list with 3 arrays, containing the edges
            print(dosemap.shape)
    #########################################################################################

            # Umrechnen in Gray (Gy = J/kg und eV = e* J )
            dosemap = dosemap*(1e6)*elementarladung / WATERMASS_PER_VOXEL # from MeV
            # Save current dosemap
            # Construct the save_filename based on the file name and position values
            filename = '{}_{}_{}_{}.npy'.format(ENERGY,file.split(".")[0].split("_")[-1],positiony,positionz)
            file_path = os.path.join(save_path, filename)
            #np.save(file_path, dosemap)
            print('{} file added successfully'.format(filename))

            # Add current dosemap to the overall dosemap
            dosemap_sum += dosemap
        
        # Save the overall dose map        
        np.save(output_path, dosemap_sum)
        print('{} file saved successfully'.format(filename_sum))

if __name__ == "__main__":
    main(sys.argv[1:])
