"""
Contains functions for training and validating a model.
"""
import torch
from model import gradient_penalty
from typing import Tuple
from utils import plot_delta, show_tensor_images
import gc

def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()
   print(f"{torch.cuda.memory_allocated()/(1024)} Kb")


def cal_passing_rate(real,fake):
    """Returns:
        A list of passing rate of individual samples in the batch 
    """
    delta = torch.abs((fake - real) / (real.max())) #δ = (Dgen − Dsim)/ Dmax sim
    #number of voxel with delta < 1%
    passing_voxels = torch.sum(delta < 0.05, dim=(1, 2, 3, 4)).tolist()  #TEST! number of voxel with delta < 5%
    batch_passing_rates = [round((passing_voxel * 100) / (256 * 256 * 128) ,2) for passing_voxel in passing_voxels] #percent passing rate
    return batch_passing_rates

def train_step(gen,
               critic,
               train_loader: torch.utils.data.DataLoader, 
               opt_gen: torch.optim.Optimizer,
               opt_critic: torch.optim.Optimizer,
               LAMBDA_GP,
               density,
               device: torch.device ):
               #writer_real,
               #writer_fake, 
               #step_real,
               #step_fake               
               
    
    """Trains a PyTorch model for a single epoch.
    Turns a model to training mode and then
    runs through all of the required training steps.

    Args:
        gen: A PyTorch generator model to be trained.
        critic: A PyTorch critic model to be trained.
        train_loader: A DataLoader instance for the model to be trained on.
        opt_gen: An optimizer to help minimize the loss function of generator.
        opt_critic: An optimizer to help minimize the loss function of discriminator.
        LAMBDA_GP : Lambda for gradient penalty
        device: A target device to compute on (e.g. "cuda" or "cpu")
        writer_real: A tensorboard writer to save real data plots
        writer_fake: A tensorboard writer to save fake data plots
        step_real: Step to see the progression of real data plots
        step_fake: Step to see the progression of fake data plots

    Returns:
        Training loss per epoch and passing rate 1%       
        (epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate_1).
    """
    report_gpu()
    # Put model in train mode
    gen.train()
    critic.train()
    # Initialize train performance metrics values
    generator_losses  = []
    critic_losses = []
    passing_rates = []

    CRITIC_ITERATIONS =5 # Update critics = CRITIC_ITERATIONS times before update the generator

    # Loop through data loader data batches
    for batch_idx, (real, cond) in enumerate(train_loader): 
        print(f"Processing train batch {batch_idx}")
        cur_batch_size = real.shape[0]    
        # Send data to target device
        # Use .float() because of RuntimeError: expected scalar type Double but found Float
        real = real.to(device) 
        print('real shape :', real.shape) 
        cond = torch.cat([density.expand(cur_batch_size, -1, -1, -1, -1), cond], dim=1) 
        cond = cond.to(device) 
        print('cond shape :', cond.shape) 
        

                     
        ########## Train Critic: max E[critic(real)] - E[critic(fake)]###########
        ########## equivalent to minimizing the negative of that #################
        # Update critics = CRITIC_ITERATIONS times before update the generator
        mean_iteration_critic_loss = 0
        for _ in range(CRITIC_ITERATIONS): 
            report_gpu()
            # print('Training Critic')
            fake = gen(cond)
            
            #print('fake shape :', fake.shape)  
            #print('cond shape :', cond.shape) 
                           
            critic_fake = critic(fake.detach(), cond).reshape(-1)  
            print("critic_fake ",critic_fake)
            critic_real = critic(real, cond).reshape(-1)   
            print("critic_real ",critic_real)  
            gp = gradient_penalty(critic, real, fake.detach(), cond, device=device)                
           
            # Calculate  and accumulate loss
            loss_critic = -(torch.mean(critic_real)- torch.mean(critic_fake)) + LAMBDA_GP*gp
            
            print("loss_critic :",-float(torch.mean(critic_real)- torch.mean(critic_fake)) ,"gradient penalty :",float(LAMBDA_GP*gp))
            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += loss_critic.item() / CRITIC_ITERATIONS
            # Optimizer zero grad to zero out any previously accumulated gradients
            critic.zero_grad()
            # Perform backpropagation
            loss_critic.backward(retain_graph=True) # True, able to reuse
            '''
            #Weight clippling # add this to restrict the gradients to prevent them from becoming too large
            if LAMBDA_GP ==0:
                for p in critic.parameters():
                    p.data.clamp_(-0.001, 0.001)
            '''
            # Optimizer step to update model parameters
            opt_critic.step()       
        critic_losses += [mean_iteration_critic_loss]
            

        ########## Train Generator: min -E[critic(gen_fake)] ##########
        ###############################################################
        #print('Training Generator')
        # Optimizer zero grad
        gen.zero_grad()
        fake_2 = gen(cond).float().to(device)
        crit_fake_pred = critic(fake_2, cond).reshape(-1)
        
        del cond
        print("crit_fake_pred ",crit_fake_pred)
        loss_gen = -1.*torch.mean(crit_fake_pred)
        #loss_gen = torch.mean(critic_real)- torch.mean(critic_fake)   
        # Loss backward
        loss_gen.backward()
        # Optimizer step
        opt_gen.step()
        # Keep track of the average generator loss
        generator_losses  += [loss_gen.item()]

        ################### Performance metric ######################
        print('calculating passing rate...')
        batch_passing_rates = cal_passing_rate(real,fake_2)
        del fake_2
       # Keep track of the passing_rate
        passing_rates += batch_passing_rates


        # Print losses occasionally and print to tensorboard
        print(f" Batch {batch_idx}/{len(train_loader)} \
                    Loss critic: {mean_iteration_critic_loss:.4f}, loss generator: {loss_gen:.4f}")           
        print(f"Train passing rate(1%) :{batch_passing_rates}") 
        #if True: #batch_idx == 0: # To change to some number
            #with torch.no_grad():
                #print(f'add image to tensor board for step {step_real}')
                #step_real = plot_dosemap(real, writer_real, step_real)# step to see the progression
                #step_fake = plot_dosemap(fake_2, writer_fake, step_fake)
                
                #show_tensor_images(real,'/home/tappay01/test/runs/images_real')
                #show_tensor_images(fake_2,'/home/tappay01/test/runs/images_fake')
                #plot_delta(real, fake_2, '/home/tappay01/test/runs/delta')

    # Calculate loss per epoch
    epoch_loss_gen = sum(generator_losses)/len(generator_losses)
    epoch_loss_critic = sum(critic_losses)/len(critic_losses)
    epoch_passing_rate = sum(passing_rates)/len(passing_rates)
    #writer_real.close()
    #writer_fake.close()

    return epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate #, step_real, step_fake

###############################End of: def train_step ####################################################
    
def val_step(gen,
               critic,
               val_loader: torch.utils.data.DataLoader, 
               density,
               device: torch.device               
               ):
    
    """Test the model for a single epoch.
    Turns a target model to "eval" mode and then performs a forward pass on a validation dataset.

    Args:
        gen: A generator model to be validated.
        critic: A critic model to be validated.
        val_loader: A DataLoader instance for the model to be validated on.
        device: A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
    Validation loss per epoch and passing rate 1%.
    """
    # Put model in eval mode
    gen.eval() 
    critic.eval()
     
    # Setup train loss values
    passing_rates = []
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop over the validation set
        for batch_idx, (real, cond) in enumerate(val_loader):
            cur_batch_size = real.shape[0]    
            # print(f"Processing val batch {batch_idx}")
            # send the input to the device
            real = real.to(device) 
            print('real shape :', real.shape) 
            cond = torch.cat([density.expand(cur_batch_size, -1, -1, -1, -1), cond], dim=1) 
            cond = cond.to(device) 
            print('cond shape :', cond.shape) 
  
            # Forward pass
            fake = gen(cond)

            print('calculating passing rate...')
            batch_passing_rates = cal_passing_rate(real, fake)
            # Keep track of the passing_rate
            passing_rates += batch_passing_rates

            # Print passing rate occasionally and # to do ...print to tensorboard
            print(f"Batch {batch_idx}/{len(val_loader)}") 
            print(f"Val passing rate 1% :{batch_passing_rates}") 
                      

    # Adjust metrics to get average passing rate per epoch
    epoch_passing_rate = sum(passing_rates)/len(passing_rates)

    return epoch_passing_rate

###############################End of: def val_step ####################################################
 