import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import matplotlib.gridspec as gridspec
from datetime import datetime


def update(results:dict, epoch_loss_gen, epoch_loss_critic, epoch_passing_rate, val_passing_rate):
    """Update result from training for each epoch.

    Append following parameters to results dictionary

    Args:
        results: A dictionary to store performance parameters
        epoch_loss_gen: train generator loss
        epoch_loss_critic: train critic loss
        epoch_passing_rate: 1% passing rate on train dataset
        val_passing_rate: 1% passing rate on validation dataset

    Returns:
        A dictionary of loss and passing rate metrics.
        
    """
    # Update results dictionary
    results["epoch_loss_gen"].append(epoch_loss_gen)
    results["epoch_loss_critic"].append(epoch_loss_critic)
    results["epoch_passing_rate"].append(epoch_passing_rate)
    results["val_passing_rate"].append(val_passing_rate)
 
    # Return the filled results at the end of the epochs
    return results

def to_voxel(position, axis):
    '''find voxel number as integer from position in mm. 
    Axis number representation 
    0 = z, 1 = y, 2 = x (beam)
    '''
    DELTA_Z = 2.5
    DELTA_X = 0.707031*2
    DELTA_Y = 0.707031*2
    if axis ==0:
        voxel = (position + 229) /DELTA_Z
    if axis ==1:
        voxel = (position + 181) /DELTA_Y
    if axis ==2:
        voxel = (position + 181) /DELTA_X
    return int(voxel) 

def extract_from_filename(filename, param):
    """Extracts energy value from the given filename."""
    parts = filename.split('_')
    if param == 'y':
        return float(parts[2][:-2])
    if param == 'z':
        return float(parts[3][:-6])
    if param == 'energy':
        return parts[1]
    else:
        return None
    

def get_original_shape(cropped_data, filename):
    # Initialize the original tensor with zeros
    original_data = np.zeros((126, 256, 256))

    y = to_voxel(extract_from_filename(filename, 'y'), 1)
    z = to_voxel(extract_from_filename(filename, 'z'), 0)
    # Calculate the starting points
    start_x = 60
    start_y = y - 8
    start_z = z - 8
    
    # Calculate the ending points
    end_x = start_x + 128
    end_y = start_y + 16
    end_z = start_z + 16

    # Assign values directly using tensor indexing
    original_data[start_z:end_z, start_y:end_y, start_x:end_x] = cropped_data
    return original_data

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_data(batch_data, batch_fake, data_name, save_folder_path):
    """Plot sample data in each batch.

    Args:
        batch_data: A batch of real data to plot.
        batch_fake: A batch of generated (fake) data to plot.
        data_name: List of names for each plot.
        save_folder_path: Location to save images.
    """
    # Get current time as a formatted string
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    # Construct a unique filename for each plot with the timestamp
    save_filename = f'plot_{timestamp}.png'
    n = 10  # Number of samples to plot
    VOXELNUMBER_X, VOXELNUMBER_Y, VOXELNUMBER_Z = 128, 16, 16

    x_vals = np.arange(VOXELNUMBER_X)
    y_vals = np.arange(VOXELNUMBER_Y)
    z_vals = np.arange(VOXELNUMBER_Z)
    index_list = []
    # Plot for real data
    fig, axs = plt.subplots(n, 3, figsize=(15, 5 * n))
    for i in range(n):
        dose = batch_data[i].squeeze().cpu().numpy()
        max_index = np.unravel_index(dose.argmax(), dose.shape)
        index_list.append(max_index)
        axs[i, 0].plot(z_vals, dose[:, int(max_index[1]), int(max_index[2])])
        axs[i, 1].plot(y_vals, dose[int(max_index[0]), :, int(max_index[2])])
        axs[i, 2].plot(x_vals, dose[int(max_index[0]), int(max_index[1]), :])

        for j in range(3):
            axs[i, j].set_yscale('linear')
            axs[i, j].set_ylabel('Dose [Gy]')
            axs[i, j].set_ylim(0.0, 1.0)

        axs[i, 0].set_xlabel('Z [mm]')
        axs[i, 1].set_xlabel('Y [mm]')
        axs[i, 2].set_xlabel('X [mm] (Beam direction)')
        axs[i, 0].set_title(f'Sample {i+1}: {data_name[i]}')
        axs[i, 2].set_title(f'Voxel : {max_index}')

    plt.suptitle('Real Data')
    plt.savefig(os.path.join(save_folder_path+'real', save_filename))
    plt.close(fig)

    # Plot for fake data
    fig, axs = plt.subplots(n, 3, figsize=(15, 5 * n))
    for i in range(n):
        dose = batch_fake[i].squeeze().cpu().numpy()
        max_index = index_list[i]
        axs[i, 0].plot(z_vals, dose[:, int(max_index[1]), int(max_index[2])])
        axs[i, 1].plot(y_vals, dose[int(max_index[0]), :, int(max_index[2])])
        axs[i, 2].plot(x_vals, dose[int(max_index[0]), int(max_index[1]), :])

        for j in range(3):
            axs[i, j].set_yscale('linear')
            axs[i, j].set_ylabel('Dose [Gy]')
            axs[i, j].set_ylim(0.0, 1.0)

        axs[i, 0].set_xlabel('Z [mm]')
        axs[i, 1].set_xlabel('Y [mm]')
        axs[i, 2].set_xlabel('X [mm] (Beam direction)')
        axs[i, 0].set_title(f'Sample {i+1}: {data_name[i]}')
        axs[i, 2].set_title(f'Voxel : {max_index}')

    plt.suptitle('Generated Data Doses')
    plt.savefig(os.path.join(save_folder_path+'fake', save_filename))
    plt.close(fig)

# Example usage:
# plot_data(batch_data, batch_fake, data_name, 'path/to/save_folder')


def plot_slice(batch_data, batch_fake, batch_density, data_name, save_folder_path):
    n = 10  # Number of samples to plot
    # Determine global min and max values for the entire batch_data
    data_max = np.max([sample.max() for sample in batch_data])

    # Common settings for colormap
    colormap_setting = {'cmap': 'inferno', 'alpha': 0.7, 'vmin': 0, 'vmax': data_max}
    # Get current time as a formatted string
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    # Construct a unique filename for each plot with the timestamp
    save_filename = f'slice_{timestamp}.png'
    index_list = []
    # Plot for real data
    fig, axs = plt.subplots(n, 3, figsize=(15, 5 * n))

    for i in range(n):
        data_sample = batch_data[i].squeeze().cpu().numpy()
        density_sample = batch_density[i].squeeze().cpu().numpy()   
        max_index = np.unravel_index(data_sample.argmax(), data_sample.shape)
        index_list.append(max_index)
        # ZY plane
        axs[i, 0].imshow(density_sample[:, :, max_index[2]], cmap='gray')
        axs[i, 0].imshow(data_sample[:, :, max_index[2]], **colormap_setting)
        axs[i, 0].set_xlabel('Z voxel number')
        axs[i, 0].set_ylabel('Y voxel number')
        axs[i, 0].set_title(data_name[i] )

        # ZX plane
        axs[i, 1].imshow(density_sample[:, max_index[1], :], cmap='gray')
        axs[i, 1].imshow(data_sample[:, max_index[1], :], **colormap_setting)
        axs[i, 1].set_xlabel('X voxel number')
        axs[i, 1].set_ylabel('Z voxel number')

        # YX plane
        axs[i, 2].imshow(density_sample[max_index[0], :, :], cmap='gray')
        img_data = axs[i, 2].imshow(data_sample[max_index[0], :, :], **colormap_setting)
        axs[i, 2].set_xlabel('X voxel number')
        axs[i, 2].set_ylabel('Y voxel number')

        # Add colorbar only on the last row
        if i == n - 1:
            fig.colorbar(img_data, ax=axs[i, :], fraction=0.046, pad=0.04).set_label('dose / dose_max')

    plt.suptitle('Real Data Slices')
    plt.savefig(os.path.join(save_folder_path+'real', save_filename))
    plt.close(fig)

    # Plot for fake data
    fig, axs = plt.subplots(n, 3, figsize=(15, 5 * n))
    
    for i in range(n):
        data_sample = batch_fake[i].squeeze().cpu().numpy()
        density_sample = batch_density[i].squeeze().cpu().numpy()
        max_index = index_list[i] # same view with real data
        # ZY plane
        axs[i, 0].imshow(density_sample[:, :, max_index[2]], cmap='gray')
        axs[i, 0].imshow(data_sample[:, :, max_index[2]],  **colormap_setting)
        axs[i, 0].set_xlabel('Z voxel number')
        axs[i, 0].set_ylabel('Y voxel number')
        axs[i, 0].set_title(data_name[i])

        # ZX plane
        axs[i, 1].imshow(density_sample[:, max_index[1], :], cmap='gray')
        axs[i, 1].imshow(data_sample[:, max_index[1], :],  **colormap_setting)
        axs[i, 1].set_xlabel('X voxel number')
        axs[i, 1].set_ylabel('Z voxel number')

        # YX plane
        axs[i, 2].imshow(density_sample[max_index[0], :, :], cmap='gray')
        img_data = axs[i, 2].imshow(data_sample[max_index[0], :, :], **colormap_setting)
        axs[i, 2].set_xlabel('X voxel number')
        axs[i, 2].set_ylabel('Y voxel number')
        # Add colorbar only on the last row
        if i == n - 1:
            fig.colorbar(img_data, ax=axs[i, :], fraction=0.046, pad=0.04).set_label('dose / dose_max')

    plt.suptitle('Generated Data Slices')
    plt.savefig(os.path.join(save_folder_path+'fake', save_filename))
    plt.close(fig)

# Example usage:
# plot_slice(batch_data, batch_fake, batch_density, data_name, 'path/to/save_folder')


def plot_delta(batch_data, batch_fake, batch_density, data_name, save_folder_path):

    # Shape of voxel
    VOXELNUMBER_X = 256
    DELTA_X = 0.707031*2
    x_vals = np.arange(-181, 181-DELTA_X, DELTA_X)
    x_vals = x_vals + DELTA_X/2

    #source_z = int(max_index[2])

    esim = get_original_shape(batch_esim[0].squeeze().numpy(), data_name[0]) # to get first sample in the batch as 3D shape and convert to numpy array

    max_index = np.unravel_index(esim.argmax(), esim.shape)
    egen = get_original_shape(batch_egen[0].squeeze().numpy(), data_name[0]) # to get first sample in the batch as 3D shape and convert to numpy array
    egen_1d = egen[ max_index[0], max_index[1], :]
    esim_1d = esim[ max_index[0], max_index[1], :]
    #Calculate delta 
    delta = ((egen_1d / esim_1d) - 1) * 100
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot Egen and Esim on the first plot
    ax1.plot(x_vals, egen_1d, 'b-', label='Generated')
    ax1.plot(x_vals, esim_1d, 'g-', label='Simulated')
    ax1.set_ylabel('Dose')
    ax1.legend(loc='upper right')

    # Plot delta on the second plot
    ax2.plot(x_vals, delta, 'r-', label='Delta')
    ax2.set_xlabel('Depth (mm)')
    ax2.set_ylabel(' ∆[%]')
    ax2.set_ylim(-10, 10)

    plt.suptitle('Comparisons of normalized simulated and generated energy deposition')
    plt.tight_layout()
 

    # Save the figure as an image (e.g., PNG)
    filename = f'{next_number}.png'
    fig.savefig(os.path.join(folder_path, filename))

    # Close the figure to release resources (optional)
    plt.close(fig)    


    


