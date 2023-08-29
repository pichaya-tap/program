"""
Contains functions for training and validating a model.
"""
import torch
from model import gradient_penalty
from typing import Tuple
from utils import plot_dosemap
import gc

def train_step(gen,
               critic,
               train_loader: torch.utils.data.DataLoader, 
               opt_gen: torch.optim.Optimizer,
               opt_critic: torch.optim.Optimizer,
               LAMBDA_GP,
               device: torch.device, 
               writer_real,
               writer_fake, 
               step_real,
               step_fake               
               ):
    
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
    gc.collect()
    # Setup train performance metrics values
    sum_loss_gen = 0
    sum_loss_critic =0
    sum_passing_rate_1 =0

    CRITIC_ITERATIONS =1 # Update critics = CRITIC_ITERATIONS times before update the generator

    # Loop through data loader data batches
    for batch_idx, (real, cond) in enumerate(train_loader): 
        print(f"Processing train batch {batch_idx}")
        # Send data to target device
        # Use .float() because of RuntimeError: expected scalar type Double but found Float
        real = real.float().to(device) 
        cond = cond.float().to(device) 
        cur_batch_size = real.shape[0]

                     
        ########## Train Critic: max E[critic(real)] - E[critic(fake)]###########
        ########## equivalent to minimizing the negative of that #################
        # Update critics = CRITIC_ITERATIONS times before update the generator
        for _ in range(CRITIC_ITERATIONS): 
            print('Training Critic')
            noise = torch.randn(cur_batch_size, 100, 16, 16, 8).float().to(device) 
            # Forward pass
            critic_real = critic(real, cond).reshape(-1)
            fake = gen(cond, noise)            
            critic_fake = critic(fake, cond).reshape(-1)
            gp = gradient_penalty(critic, real, fake, cond, device=device)
            # Calculate  and accumulate loss
            loss_critic = (
                -(torch.mean(critic_real)- torch.mean(critic_fake)) + LAMBDA_GP*gp
                )
            # Optimizer zero grad
            critic.zero_grad()
            # Loss backward
            loss_critic.backward(retain_graph=True) # able to reuse
            # Optimizer step
            opt_critic.step() 
            # Accumulate metric across all interations
            sum_loss_critic += float(loss_critic)
            

        ########## Train Generator: min -E[critic(gen_fake)] ##########
        ###############################################################
        print('Training Generator')
        #noise = torch.randn(cur_batch_size, Z_DIM, 16, 16,8).to(device)
        #fake = gen(cond, noise) #reuse the fake tensor
        output = critic(fake, cond).reshape(-1)
        loss_gen = -torch.mean(output)
        # Optimizer zero grad
        gen.zero_grad()
        # Loss backward
        loss_gen.backward()
        # Optimizer step
        opt_gen.step()
        # Accumulate metric across all batches
        sum_loss_gen += float(loss_gen)

        ################### Performance metric ######################
        print('calculating passing rate...')
        delta = torch.abs((fake - real) / real.max()) #δ = (Dgen − Dsim)/ Dmax sim
        passing_voxel_1 = torch.sum(delta < 0.01).item() #number of voxel with delta < 1%
        passing_rate_1 = passing_voxel_1*100/(256*256*128) #percent passing rate

        # Accumulate metric across all batches
        sum_passing_rate_1 += passing_rate_1


        # Print losses occasionally and print to tensorboard
        if batch_idx == 0: # To change to some number
            print(f"Batch {batch_idx}/{len(train_loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")           
            print(f"Train passing rate(1%) :{passing_rate_1:.2f}") 

            with torch.no_grad():
                print(f'add image to tensor board for step {step_real}')
                step_real = plot_dosemap(real, writer_real, step_real)# step to see the progression
                step_fake = plot_dosemap(fake, writer_fake, step_fake)
      

    # Calculate loss per epoch
    epoch_loss_gen = sum_loss_gen/len(train_loader)
    epoch_loss_critic = sum_loss_critic/ (len(train_loader)*CRITIC_ITERATIONS)
    epoch_passing_rate_1 = sum_passing_rate_1/len(train_loader)

    writer_real.close()
    writer_fake.close()

    return epoch_loss_gen,  epoch_loss_critic, epoch_passing_rate_1, step_real, step_fake

###############################End of: def train_step ####################################################
    
def val_step(gen,
               critic,
               val_loader: torch.utils.data.DataLoader, 
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
    sum_passing_rate_1 = 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch_idx, (real, cond) in enumerate(val_loader): 
            print(f"Processing val batch {batch_idx}")
            real = real.float().to(device) #RuntimeError: expected scalar type Double but found Float
            cond = cond.float().to(device) #RuntimeError: expected scalar type Double but found Float
            cur_batch_size = real.shape[0]
            noise = torch.randn(cur_batch_size, 100, 16, 16,8).float().to(device)
            # Forward pass
            fake = gen(cond, noise)

            # Calculate and accumulate passing rate
            delta =(fake - real)/ real.max() #δ = (Dgen − Dsim)/ Dmax sim
            passing_voxel_1 = torch.sum(delta < 0.01).item()
            passing_rate_1 = passing_voxel_1*100/(256*256*128)

            # Accumulate metric across all batches
            sum_passing_rate_1 += passing_rate_1

            # Print passing rate occasionally and # to do ...print to tensorboard
            print(
                f"Batch {batch_idx}/{len(val_loader)} \
                    Val passing rate 1%: {passing_rate_1:.4f}"
            )
                      

    # Adjust metrics to get average passing rate per epoch
    epoch_passing_rate_1 = sum_passing_rate_1/len(val_loader)

    return epoch_passing_rate_1

###############################End of: def val_step ####################################################
 