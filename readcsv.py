import numpy as np
import pandas as pd
import multiprocessing
import time
import pandas as pd
import time
import dask.dataframe as dd


file = '/gpfs001/scratch/schwar14/simudosemapheavyions/Watersimulation/Hadr01/Dataset_Miriam/2750MeV/output_nt_Hits_t1.csv'

print('reading file')

# Record the start time
start_time = time.time()

data = pd.DataFrame()  # Create an empty DataFrame to store the data

# Set the chunk size (adjust as needed)
chunk_size = 10000

for chunk in pd.read_csv(file, sep=',', skiprows=11, names=['X', 'Y', 'Z', 'Edep', 'Process', 'particle', 'E'], chunksize=chunk_size):
    # Append the chunk to the data DataFrame
    data = pd.concat(chunk, ignore_index=True)
    
    # Print a progress message
    print(f'Reading {len(data)} rows...', end='\r')

# Record the end time
end_time = time.time()

# Calculate the elapsed time in seconds
elapsed_time = end_time - start_time

print('done')
print(f'Time taken to read the file: {elapsed_time:.2f} seconds')
