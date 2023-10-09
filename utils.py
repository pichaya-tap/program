import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

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

def show_tensor_images(batch_data, folder_path):
    """Plot sample data in each batch
   
    Args:
        batch_data: A batch data to plot in the shape of [BATCHSIZE*1*256*256*128]
        folder: location to save images
    """

    # Shape of voxel
    batch_size = batch_data.shape[0]
    VOXELNUMBER_X = 256
    VOXELNUMBER_Y = 256
    VOXELNUMBER_Z = 126

    DELTA_Z = 2.5
    DELTA_X = 0.707031*2
    DELTA_Y = 0.707031*2

    x_vals = np.arange(-181, 181-DELTA_X, DELTA_X)
    x_vals = x_vals + DELTA_X/2

    y_vals = np.arange(-181, 181-DELTA_Y, DELTA_Y)
    y_vals = y_vals + DELTA_Y/2

    z_vals = np.arange(-200, 115, DELTA_Z)
    #z_vals = np.arange(-229,86, DELTA_Z)
    z_vals = z_vals + DELTA_Z/2

    # Create a batchx3 grid of subplots
    fig, axs = plt.subplots(batch_size, 3, figsize=(10, 8))# plot only 1 row

    
    #source_y = int(max_index[1]) # it is not exact as source position
    #source_z = int(max_index[2])
    for i in range(batch_size):
        dose = batch_data[i].squeeze().cpu().numpy() # from first sample data in batch to get 3D shape and convert to numpy array
        dose = get_original_shape(dose)
        max_index = np.unravel_index(dose.argmax(), dose.shape)

        dose_x = []
        dose_y = []
        dose_z = []
        for x in range(VOXELNUMBER_X):
            dose_x.append(dose[x, int(max_index[1]),int(max_index[2])])
        for y in range(VOXELNUMBER_Y):
            dose_y.append(dose[int(max_index[0]),y,int(max_index[2])])
        for z in range(VOXELNUMBER_Z):
            dose_z.append(dose[int(max_index[0]), int(max_index[1]),z])

        axs[i,0].plot(x_vals, dose_x)
        axs[i,0].set_yscale('linear')
        axs[i,0].set_xlabel('X [mm]')
        axs[i,0].set_ylabel('Normalized Dose')

        axs[i,1].plot(y_vals, dose_y)
        axs[i,1].set_yscale('linear')
        axs[i,1].set_xlabel('Y [mm]')

        axs[i,2].plot(z_vals, dose_z)
        axs[i,2].set_yscale('linear')
        axs[i,2].set_xlabel('Z [mm]')

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Find the highest numbered file in the folder
    existing_files = os.listdir(folder_path)
    existing_numbers = []

    for filename in existing_files:
        if filename.endswith('.png'):
            try:
                number = int(filename.split('.')[0])
                existing_numbers.append(number)
            except ValueError:
                pass

    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        next_number = 0
    # Save the figure as an image (e.g., PNG)
    filename = f'{next_number}.png'
    fig.savefig(os.path.join(folder_path, filename))

    # Close the figure to release resources (optional)
    plt.close(fig)    

def get_original_shape(smaller_tensor):
    # Define the original shape
    original_shape = (256, 256, 126)
    # Create a tensor of zeros with the original shape
    original_tensor = np.zeros(original_shape)

    # Define the slice to insert the smaller tensor
    slice_x = slice(60, 188)
    slice_y = slice(75, 203)
    slice_z = slice(18, 82)

    # Insert the smaller tensor into the original tensor at the specified slice
    original_tensor[slice_x, slice_y, slice_z] = smaller_tensor
    return original_tensor
  

def plot_dosemap(batch_data, tensorboard_writer:torch.utils.tensorboard.SummaryWriter, step:int):
    """Plot one sample data in each batch

    Add a sample image to tensorboard writer
    
    Args:
        batch_data: A batch data to plot in the shape of [BATCHSIZE*1*256*256*128]
        tensorboard_writer: SummaryWriter of tensorboard
        step: An number to indicate progression 
    
    Returns:
        Increase step of 1
    """

    # Shape of voxel
    batch_size = batch_data.shape[0]
    VOXELNUMBER_X = 256
    VOXELNUMBER_Y = 256
    VOXELNUMBER_Z = 128

    DELTA_Z = 2.5
    DELTA_X = 0.707031*2
    DELTA_Y = 0.707031*2

    x_vals = np.arange(-181, 181-DELTA_X, DELTA_X)
    x_vals = x_vals + DELTA_X/2

    y_vals = np.arange(-181, 181-DELTA_Y, DELTA_Y)
    y_vals = y_vals + DELTA_Y/2

    z_vals = np.arange(-200, 120, DELTA_Z)
    z_vals = z_vals + DELTA_Z/2

    # Create a batchx3 grid of subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))# plot only 1 row

    dose = batch_data[0].squeeze().cpu().numpy() # from first sample data in batch to get 3D shape and convert to numpy array
    max_index = np.unravel_index(dose.argmax(), dose.shape)

    #source_y = int(max_index[1]) # it is not exact as source position
    #source_z = int(max_index[2])

    dose_x = []
    dose_y = []
    dose_z = []
    for x in range(VOXELNUMBER_X):
        dose_x.append(dose[x, int(max_index[1]),int(max_index[2])])
    for y in range(VOXELNUMBER_Y):
        dose_y.append(dose[int(max_index[0]),y,int(max_index[2])])
    for z in range(VOXELNUMBER_Z):
        dose_z.append(dose[int(max_index[0]), int(max_index[1]),z])
    axs[0].plot(x_vals, dose_x)
    axs[0].set_yscale('linear')
    axs[0].set_xlabel('X [mm]')
    axs[0].set_ylabel('E/Emax')

    axs[1].plot(y_vals, dose_y)
    axs[1].set_yscale('linear')
    axs[1].set_xlabel('Y [mm]')

    axs[2].plot(z_vals, dose_z)
    axs[2].set_yscale('linear')
    axs[2].set_xlabel('Z [mm]')

    # Render the plot as an image using FigureCanvasAgg
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Convert the canvas to a numpy array
    img_array = np.array(canvas.buffer_rgba())
    # Convert the image array to tensor format and add to TensorBoard
    tensor_img = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    tensorboard_writer.add_image("Line_Plot", tensor_img, dataformats="CHW", global_step=step)
    
    plt.close(fig)  # Close the figure after using it
    return step + 1 


def plot_delta(batch_esim,batch_egen, folder_path):

    # Shape of voxel
    VOXELNUMBER_X = 256
    DELTA_X = 0.707031*2
    x_vals = np.arange(-181, 181-DELTA_X, DELTA_X)
    x_vals = x_vals + DELTA_X/2

    #source_z = int(max_index[2])

    esim = get_original_shape(batch_esim[0].squeeze().numpy()) # to get first sample in the batch as 3D shape and convert to numpy array

    max_index = np.unravel_index(esim.argmax(), esim.shape)
    egen = get_original_shape(batch_egen[0].squeeze().numpy()) # to get first sample in the batch as 3D shape and convert to numpy array
    egen_1d = egen[:, max_index[1], max_index[2]]
    esim_1d = esim[:, max_index[1], max_index[2]]
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
 

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Find the highest numbered file in the folder
    existing_files = os.listdir(folder_path)
    existing_numbers = []

    for filename in existing_files:
        if filename.endswith('.png'):
            try:
                number = int(filename.split('.')[0])
                existing_numbers.append(number)
            except ValueError:
                pass

    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        next_number = 0
    # Save the figure as an image (e.g., PNG)
    filename = f'{next_number}.png'
    fig.savefig(os.path.join(folder_path, filename))

    # Close the figure to release resources (optional)
    plt.close(fig)    


    


