import torch
import os
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
from datetime import datetime
from pathlib import Path
#import torchinfo    
#from torchinfo import summary
import time
from tqdm import tqdm
import pickle
import gc

from dataloader import CustomDataset
from model2 import Critic3d, Generator, initialize_weights
from engine import train_step, val_step
from utils import update
from torch.utils.data import ConcatDataset, random_split
######################################################################
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device :',device)

torch.backends.cudnn.enabled=True # comment BatchNorm1d -> CUDNN_STATUS_NOT_SUPPORTED
# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True

def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()
   print(f"{torch.cuda.memory_allocated()/(1024)} Kb")

######################### HYPERPARAMETER ##############################
LEARNING_RATE_G = 0.001 
LEARNING_RATE_C = 0.00002 
Z_DIM =100
NUM_EPOCHS = 200
# CRITIC_ITERATIONS =5 #Parameter to update critic many times before update generator once.Change im engine
LAMBDA_GP = 10 #Lambda for gradient penalty 
BATCH_SIZE = 32 #To be 32 according to paper

#######################################################################
########################## DATA LOADING ###############################
print('loading data as batch')

data_folders = [
    "/scratch/tappay01/data/data1_resampled", 
    "/scratch/tappay01/data/data2_resampled",
    "/scratch/tappay01/data/data4_resampled",
    "/scratch/tappay01/data/data5_resampled",
    "/scratch/tappay01/data/data7_resampled"
]

density_folder = "/scratch/tappay01/densities"
water_folder = "/scratch/tappay01/watersimulation/DATASET"

# dataset = CustomDataset(data_folder, density_folder, water_folder)
# Initialize datasets separately

# Create datasets for each folder and combine them into a single dataset
# combined_dataset = ConcatDataset([CustomDataset(folder, density_folder, water_folder) for folder in data_folders])

#Save the custom dataset to a file
saved_dataset = '/scratch/tappay01/custom_dataset2.pkl'
#with open(saved_dataset, 'wb') as file:
#    pickle.dump(combined_dataset, file)
#Later, when you want to use the dataset again, you can load it from the file
with open(saved_dataset, 'rb') as file:
   combined_dataset = pickle.load(file)

print('Total data :',len(combined_dataset))
# Split combined dataset into train, validation, and test
train_size = int(0.7 * len(combined_dataset))
valid_size = int(0.3 * len(combined_dataset))
test_size = len(combined_dataset) - (train_size + valid_size)

train_subset, val_subset, test_subset = random_split(combined_dataset, [train_size, valid_size, test_size])

# Turn train, val and test custom Dataset into DataLoader's
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8) #creates 8 worker processes to load the data in parallel. 
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

print("Total number of samples in the train dataset:", len(train_subset))
print("Number of batches:", len(train_loader))
print("Total number of samples in the validation dataset:", len(val_subset))
print("Number of batches:", len(val_loader))
print("Total number of samples in the test dataset:", len(test_subset))
print("Number of batches:", len(test_loader))

#######################################################################
######################### CREATE LOG DIRECTORY ########################
# Tensorboard set up
log_dir = "/home/tappay01/test/runs/"+datetime.now().strftime("%d%m%H%M")+"/"


# Create target directory
target_dir_path = Path(log_dir)
target_dir_path.mkdir(parents=True,exist_ok=True)
# Create the 'fake' directory
fake_dir_path = target_dir_path / 'fake'
fake_dir_path.mkdir(parents=True, exist_ok=True)

# Create the "real" directory under log_dir
real_dir_path = Path(log_dir) / 'real'
real_dir_path.mkdir(parents=True, exist_ok=True)

#######################################################################
########################### INITIALIZE MODELS ##########################
gen = Generator()
initialize_weights(gen)
gen = gen.to(device)
critic = Critic3d()
initialize_weights(critic)
critic = critic.to(device)
for param in gen.parameters():
    param= param.to(device)
for param in critic.parameters():
    param= param.to(device)
#######################################################################
################################ TRAIN MODELS ##########################
#Set up optimizers for generator and critic
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.9)) 
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_C, betas=(0.0, 0.9))

# Create empty results dictionary
results = {"epoch_loss_gen": [],
            "epoch_loss_critic": [],
            "epoch_passing_rate": [],
            "val_passing_rate": [], 
}
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)): 
    #report_gpu()
    #epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate, step_real, step_fake   
    epoch_loss_gen,epoch_loss_critic,epoch_passing_rate = train_step(gen, critic,train_loader,opt_gen,opt_critic,LAMBDA_GP, device,log_dir) 
                                                      
                                                    
    
    #print('End of training step. Start validation step')
    val_passing_rate = val_step(gen, critic, val_loader, device)

    # Update results dictionary
    results = update(results, epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate, val_passing_rate)

    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss generator: {:.6f}, Train loss critic: {:.4f}, Train passing rate: {:.4f}".format(
		epoch_loss_gen, epoch_loss_critic, epoch_passing_rate))

    # Save Model when new high validation passing rate is found
    if all(val_passing_rate > rate for rate in results["val_passing_rate"]):
        print(f"[INFO] Found new high passing rate. Saving model to: {log_dir}")
        model_name = f"highest passing rate model_epoch{e}.pth"
        # Save the model state_dict()
        torch.save(obj=gen.state_dict(),f=target_dir_path/model_name) 
    
model_name = f"model_epoch{e}.pth"
# Save the model state_dict()
torch.save(obj=gen.state_dict(),f=target_dir_path/model_name) 



# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

#######################################################################
####################### VISUALIZE RESULTS ############################

# Plot loss per epoch
plt.figure(figsize=(10,5))
plt.title("Generator and Critic Loss During Training Per Epoch ")
plt.plot(results["epoch_loss_gen"],label="Generator_loss")
plt.plot(results["epoch_loss_critic"],label="Critic")
plt.xlabel("number of epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(log_dir + "loss.png")
plt.clf()


# Plot passing rate per epoch
# Find the highest value and its index
highest_value = max(results["val_passing_rate"])
highest_index = results["val_passing_rate"].index(highest_value)
# Plot passing rate per epoch
plt.figure(figsize=(10,5))
plt.title("Passing Rate During Training")
plt.plot(results["epoch_passing_rate"], label="train_passing_rate")
plt.plot(results["val_passing_rate"], label="val_passing_rate")
# Mark the highest value
plt.scatter(highest_index, highest_value, color='red') # Mark with a red dot
plt.text(highest_index, highest_value, f' Max: {highest_value}', color='red') # Annotate the point
plt.xlabel("number of epoch")
plt.ylabel("Passing rate")
plt.legend()
plt.savefig(log_dir + "passing_rate.png")
plt.clf()

