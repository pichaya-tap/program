import torch
import torch.nn as nn

from torch.autograd import Variable

class Critic3d(nn.Module):
    """Creates the convolutional network
    
    Replicates critic network from https://arxiv.org/abs/2202.07077
    consisting of six consecutive transposed 3D convolutions.

    Returns:
        A float as continuous and quantitative rating"""
    
    def __init__(self):
        super(Critic3d, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride =2, padding =1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride =2, padding =1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout3d(p=0.15),

            nn.Conv3d(128, 256, kernel_size=3, stride =2, padding =1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 1, kernel_size=3, stride =2, padding =0, bias=False),
            nn.Flatten(),
            nn.Dropout3d(p=0.15),
            nn.Linear(63,1) #1575 when original shape
            
        )

    def forward(self, x, conditional_input ):
        x = self.conv_stack(torch.cat([x, conditional_input], dim=1))      
        # Reshape the tensor to [N, 1]
        return x  
    
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout3d(p=0.15)

    def forward(self, x):
        return self.dropout(self.silu(self.bn2(self.conv2(self.silu(self.bn1(self.conv1(x)))))))
    
       

class Encoder(nn.Module):
    def __init__(self, channels=(2, 16, 32, 64)):
        super().__init__()
        self.enc_block = nn.ModuleList(
            [Block(channels[i], channels[i+1])
                for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool3d((2,2, 2))

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []

        # loop through the encoder blocks
        for block in self.enc_block:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
            x = block(x)
            #print("encoder block",x.shape)
            block_outputs.append(x)
            x = self.pool(x)
            #print("pooling",x.shape)
		# return the list containing the intermediate outputs
        return x, block_outputs


class Decoder(nn.Module):
    def __init__(self, channels=(128, 64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.up = nn.ModuleList(
			[nn.ConvTranspose3d(channels[i], channels[i + 1], kernel_size=2, stride=2, padding=0)
			 	for i in range(len(channels) - 1)])
        self.dec_blocks = nn.ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
		# loop through the number of channels
        for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
            x = self.up[i](x)
            #print("upsampling",x.shape)
			# crop the current features from the encoder blocks,
            encFeat = self.crop(encFeatures[i], x)
			# concatenate them with the current upsampled features,
            x = torch.cat([x, encFeat], dim=1)
			# and pass the concatenated output through the current decoder block
            x = self.dec_blocks[i](x)
		# return the final decoder output
        return x
    
    def crop(self, encFeatures, x):
        # Grab the dimensions of the inputs and crop the encoder features
        # to match the spatial dimensions of x (H, W).
        (_, _, D, H, W) = x.shape

        # Assuming you want to crop the center region, you can calculate the starting
        # indices for cropping as follows:
        start_d = (encFeatures.shape[2] - D) // 2
        start_h = (encFeatures.shape[3] - H) // 2
        start_w = (encFeatures.shape[4] - W) // 2

        # Crop the encoder features using PyTorch's slicing operations.
        cropped_features = encFeatures[:, :, start_d:start_d+D, start_h:start_h+H, start_w:start_w+W]

        # Return the cropped features
        return cropped_features


class Generator(nn.Module):
    """Creates the U-Net Architecture
    
    Replicates conditional GAN with a Wasserstein loss function and a Gradient
    Penalty term from https://arxiv.org/abs/2202.07077

    Returns:
        A batch of 3D matrix of energy depositions of size [BATCHSIZE*1*256*256*128]
    """
    def __init__(self, encChannels=(2, 16, 32, 64)):
        decChannels=(128,64, 32, 16)
        super().__init__()
		# initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        """ Bottleneck """
        self.b = Block(164, 128) #add 100 dimension for noise 

        """ Classifier """
        self.outputs = nn.Sequential(
                            nn.Conv3d(16, 1, kernel_size=1, padding=0), #??? ConvTranspose3d or nn.Conv3d
                            nn.Sigmoid(), # or Tanh()?
        )

    def forward(self, x):
		# grab the features from the encoder
        b, encFeatures = self.encoder(x)
        #print("last encFeatures {}".format(encFeatures[::-1][0].shape))
        #print("bottleneck {}".format(b.shape))

        # Concatenate input tensor and noise tensor along the channel dimension (dim=1)
        add_noise = torch.cat([b, 
                               torch.randn((x.shape[0], 100, b.shape[2], b.shape[3], b.shape[4]), 
                                            device=b.device)], 
                                            dim=1)
        #print("noise {}".format(add_noise.shape))
        b = self.b(add_noise) 
        #print("bottleneck with noise {}".format(b.shape))
        # pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
        decFeatures = self.decoder(b, encFeatures[::-1][0:])

        return self.outputs(decFeatures)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
            m.weight.data = torch.randn(m.weight.data.shape)
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def gradient_penalty(critic, real, fake, cond, device):
    """Gradient Penalty to regularize and stabilize the weight updates
    
    Args:
        critic: A PyTorch critic model to be trained.
        real: A batch of train dataset in the shape of [BATCHSIZE*1*256*256*128]
        fake: A batch of output from generator in the shape of [BATCHSIZE*1*256*256*128]
        cond: Respective conditional input of the data batch [BATCHSIZE*2*256*256*128]
        device: A target device to compute on (e.g. "cuda" or "cpu")
    
    Returns:
        gradient_penalty term
    """
    # Random weight term for interpolation between real and fake samples
    epsilon = torch.rand((real.shape[0], 1, 1, 1, 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (epsilon * real + ((1 - epsilon) * fake)).requires_grad_(True)

    # Calculate gradients
    mixed_scores = critic(interpolates, cond)

    #fake = Variable(torch.Tensor(real.shape[0], 1).to(device).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        inputs=interpolates,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), #fake
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0] # get first element

    gradients = gradients.view(len(gradients), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    #slopes  = torch.sqrt(torch.sum(gradient ** 2, dim=[1, 2, 3, 4]) + 1e-6) #small eps value to avoid Nan in sqr(0)
    #return torch.mean((slopes - 1)**2) # gradient_penalty

# To test and see summary of models # uncomment below
'''
import torchinfo    
from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator(encChannels=(2, 16, 32, 64)).to(device)
initialize_weights(gen)
critic = Critic3d().to(device)
initialize_weights(critic)

noise = torch.randn((1,100,16,16,8)).to(device) #Will be fed to generator's bottle neck
summary(gen, input_size=[(1,2, 128, 64, 64)]) # do a test pass through of an example input size 
summary(critic, input_size=[(8,1, 128, 64, 64),(8,2,128,64,64)])
'''