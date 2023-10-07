import torch
import torch.nn as nn
import torchinfo    
from torchinfo import summary
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
            nn.Linear(1575,1)
            
        )


    def forward(self, x, conditional_input ):
        x = self.conv_stack(torch.cat([x, conditional_input], dim=1))      
        # Reshape the tensor to [N, 1]
        return x  
    

class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.dropout = nn.Dropout3d(p=0.15)

    def forward(self, x, batchnorm=True):
        if batchnorm:
            return self.dropout(self.silu(self.bn2(self.conv2(self.silu(self.bn1(self.conv1(x)))))))
        else:
            return self.dropout(self.silu(self.conv2(self.silu(self.conv1(x)))))

  
     

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d((2,2, 2))

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []
        # pass the inputs through the encoder blocks, store
        # the outputs, and then apply maxpooling on the output
        x = Block(2, 16)(x, batchnorm =False)
        #print("encoder block",x.shape)
        block_outputs.append(x)
        x = self.pool(x)
        x = Block(16, 32)(x)
        #print("encoder block",x.shape)
        block_outputs.append(x)
        x = self.pool(x)
        x = Block(32, 64)(x)
        #print("encoder block",x.shape)
        block_outputs.append(x)
        x = self.pool(x)
        #print("last pooling",x.shape)
		# return the list containing the intermediate outputs
        return x, block_outputs

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
        self.conv1 = Block(128,64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
        self.conv2 = Block(64,32)
        self.up3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
        self.conv3 = Block(32,16)
    def forward(self, x, skip):
		# loop through the number of channels
        #print('skip:', len(skip))
        x = self.up1(x)
        #print(x.shape, skip[0].shape)
        x = torch.cat([x, skip[0]],axis=1) #([1, 64, 64, 64, 32])
        #print(x.shape) #([1, 128, 64, 64, 32])
        x = self.conv1(x)        
        #print(x.shape) #([1, 64, 64, 64, 32])
        x = self.up2(x)
        #print(x.shape, skip[1].shape) #([1, 32, 128, 128, 64])
        x = torch.cat([x, skip[1]],axis=1) #([1, 32, 128, 128, 64])
        x = self.conv2(x)

        x = self.up3(x)
        #print(x.shape, skip[2].shape)
        x = torch.cat([x, skip[2]],axis=1) #([1, 16, 256, 256, 128])
        x = self.conv3(x)

        return x
    
class Generator(nn.Module):
    """Creates the U-Net Architecture
    
    Replicates conditional GAN with a Wasserstein loss function and a Gradient
    Penalty term from https://arxiv.org/abs/2202.07077

    Returns:
        A batch of 3D matrix of energy depositions of size [BATCHSIZE*1*256*256*128]
    """
    def __init__(self):
        super().__init__()
		# initialize the encoder and decoder
        self.encoder = Encoder()

        """ Bottleneck """
        self.bottleneck = Block(164, 128) #add 100 dimension for noise 
        
        self.decoder = Decoder()

        self.outputs = nn.Sequential(
                            nn.Conv3d(16, 1, kernel_size=1, padding=0),
                            nn.Sigmoid(), #output value in the range [0,1]
        )

    def forward(self, x):
		# grab the features from the encoder
        b, encFeatures = self.encoder(x)
        #print("last encFeatures {}".format(encFeatures[::-1][0].shape))
        #print("bottleneck {}".format(b.shape))

        # Concatenate input tensor and noise tensor along the channel dimension (dim=1)
        b= torch.cat((b, 
                    torch.randn((x.shape[0], 100, b.shape[2], b.shape[3], b.shape[4]), 
                                            device=b.device)), 
                                            dim=1)
        #print("noisy tensor {}".format(b.shape))

        b = self.bottleneck(b) 
        #print("bottleneck with noise {}".format(b.shape))

        # pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
        decFeatures = self.decoder(b, encFeatures[::-1][0:])

        return  self.outputs(decFeatures)


def initialize_weights(model, device):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
            m.weight.data = torch.randn(m.weight.data.shape).to(device)
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
    
    if torch.isnan(gradients).any():
        print("Gradients contain NaN values")

    gradients = gradients.view(len(gradients), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    #slopes  = torch.sqrt(torch.sum(gradient ** 2, dim=[1, 2, 3, 4]) + 1e-6) #small eps value to avoid Nan in sqr(0)
    #return torch.mean((slopes - 1)**2) # gradient_penalty

# To test and see summary of models # uncomment below

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
initialize_weights(gen, device)
critic = Critic3d().to(device)
initialize_weights(critic, device)

noise = torch.randn((1,100,16,16,8)).to(device) #Will be fed to generator's bottle neck
#summary(gen, input_size=[(1,2, 256, 256, 128)]) # do a test pass through of an example input size 
summary(critic, input_size=[(8,1, 256, 256, 128),(8,2,256,256,128)])
