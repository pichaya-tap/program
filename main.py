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

from dataloader import CustomDataset, split
from model import Critic3d, Generator, initialize_weights
from engine import train_step, val_step
from utils import update
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
LEARNING_RATE = 1e-4 # could also use 2 lrs
Z_DIM =100
NUM_EPOCHS = 100
# CRITIC_ITERATIONS =5 #Parameter to update critic many times before update generator once.Change im engine
# WEIGHT_CLIP = 0.01 #If use weight clipping. We use Wasserstein distance instead.
LAMBDA_GP = 10 #Lambda for gradient penalty 
BATCH_SIZE = 16 #To be 32 according to paper
print("BATCH_SIZE" ,BATCH_SIZE)

#######################################################################
########################## DATA LOADING ###############################
print('loading data as batch')

data_folder = '/home/tappay01/data/data1/' #Dataset1
water_dosemap_folder = '/home/tappay01/data/water/DATASET' #First conditional input 
density_file = "/home/tappay01/data/DATASET_densities.npy" #Second conditional input
print('create custom_dataset')

#custom_dataset = CustomDataset(data_folder, water_dosemap_folder, density_file)
#Save the custom dataset to a file
#with open('/home/tappay01/data/custom_dataset3.pkl', 'wb') as file:
 #   pickle.dump(custom_dataset, file)

#Later, when you want to use the dataset again, you can load it from the file
with open('/home/tappay01/data/custom_dataset3.pkl', 'rb') as file:
    custom_dataset = pickle.load(file)

print('# Split to train, validation, test subset')
train_subset, val_subset, test_subset = split(custom_dataset, 0.9, 0.1, 0.0)
# Turn train, val and test custom Dataset into DataLoader's
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

print("Total number of1575 samples in the train dataset:", len(train_subset))
print("Number of batches:", len(train_loader))
print("Total number of samples in the validation dataset:", len(val_subset))
print("Number of batches:", len(val_loader))
print("Total number of samples in the test dataset:", len(test_subset))
print("Number of batches:", len(test_loader))

#######################################################################
######################### CREATE LOG DIRECTORY ########################
# Tensorboard set up
log_dir = "/home/tappay01/test/runs/"+datetime.now().strftime("%m%d%H%M")+"/"
writer_loss = SummaryWriter(os.path.join(log_dir, 'loss'))
#writer_passing_rate = SummaryWriter(os.path.join(log_dir, 'passing rate'))
writer_real = SummaryWriter(os.path.join(log_dir, 'real'))
#step_real = 0 
#step_fake = 0 
# Create target directory
target_dir_path = Path(log_dir)
target_dir_path.mkdir(parents=True,exist_ok=True)


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
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)) 
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


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
    report_gpu()
    #epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate, step_real, step_fake   
    epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate = train_step(gen, critic,train_loader,opt_gen,opt_critic,LAMBDA_GP,device) 
                                                        #writer_real,
                                                        #writer_fake, 
                                                        #step_real,
                                                        #step_fake 
                                                        
    
    print('End of training step. Start validation step')
    val_passing_rate = val_step(gen, critic, val_loader, device)


    # Update results dictionary
    results = update(results, epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate, val_passing_rate)

    # Add results to SummaryWriter
    #writer_loss.add_scalars(main_tag="Loss",
                          #tag_scalar_dict={"gen_loss": epoch_loss_gen,
                           #                 "critic_loss": epoch_loss_critic},
                          #global_step=e)
    #writer_passing_rate.add_scalars(main_tag="Passing rate",
                          #tag_scalar_dict={"passing_rate_1": val_passing_rate},
                          #global_step=e)
    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss generator: {:.6f}, Train loss critic: {:.4f}, Train passing rate: {:.4f}, Val passing rate: {:.4f}".format(
		epoch_loss_gen, epoch_loss_critic, epoch_passing_rate, val_passing_rate))

    # Save Model when new high validation passing rate is found
    if all(val_passing_rate > rate for rate in results["val_passing_rate"]):
        print(f"[INFO] Found new high passing rate. Saving model to: {log_dir}")
        model_name = f"model_epoch{e}.pth"
        # Save the model state_dict()
        torch.save(obj=gen.state_dict(),f=target_dir_path/model_name) 

# Close all the writers
#writer_real.close()
#writer_fake.close()
#writer_loss.close() 
#writer_passing_rate.close()

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
plt.figure(figsize=(10,5))
plt.title("Passing Rate During Training")
plt.plot(results["epoch_passing_rate"],label="train_passing_rate")
plt.plot(results["val_passing_rate"],label="val_passing_rate")
plt.xlabel("number of epoch")
plt.ylabel("Passing rate")
plt.legend()
plt.savefig(log_dir + "passing_rate.png")
plt.clf()

