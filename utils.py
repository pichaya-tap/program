import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def update(results:dict, epoch_loss_gen, epoch_loss_critic, epoch_passing_rate_1, val_passing_rate_1):
    """Update result from training for each epoch.

    Append following parameters to results dictionary

    Args:
        results: A dictionary to store performance parameters
        epoch_loss_gen: train generator loss
        epoch_loss_critic: train critic loss
        epoch_passing_rate_1: 1% passing rate on train dataset
        val_passing_rate_1: 1% passing rate on validation dataset

    Returns:
        A dictionary of loss and passing rate metrics.
        
    """
    # Update results dictionary
    results["epoch_loss_gen"].append(epoch_loss_gen)
    results["epoch_loss_critic"].append(epoch_loss_critic)
    results["epoch_passing_rate_1"].append(epoch_passing_rate_1)
    results["val_passing_rate_1"].append(val_passing_rate_1)
    # Print out what's happening
    print("epoch_loss_gen :", epoch_loss_gen)
    print("epoch_loss_critic :", epoch_loss_critic)
    print("epoch_passing_rate_1 :", epoch_passing_rate_1)
    print("val_passing_rate_1 :", val_passing_rate_1)
 
    # Return the filled results at the end of the epochs
    return results


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

    z_vals = np.arange(-229, 91, DELTA_Z)
    z_vals = z_vals + DELTA_Z/2

    # Create a batchx3 grid of subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))# plot only 1 row

    dose = batch_data[0].squeeze().numpy() # from first sample data in batch to get 3D shape and convert to numpy array
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

