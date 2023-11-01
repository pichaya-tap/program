import os

# specify the directory containing the files
directory = '/scratch/tappay01/data/data1/'

for filename in os.listdir(directory):
    # check if the filename contains a comma
    if ',' in filename:
        new_filename = filename.replace(',', '')  # remove commas
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")
