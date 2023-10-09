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
            nn.Dropout3d(p=0.15),
            nn.Linear(1575,1)
            
        )


    def forward(self, x, conditional_input ):
        x = self.conv_stack(torch.cat([x, conditional_input], dim=1))      
        # Reshape the tensor to [N, 1]
        return x  
    



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            #nn.ZeroPad3d((1, 0, 1, 0)),
            nn.Conv3d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        b = self.down8(d7)
        # Concatenate input tensor and noise tensor along the channel dimension (dim=1)
        b = torch.cat([b, torch.randn((x.shape[0], 100, b.shape[2], b.shape[3], b.shape[4]), 
                                            device=b.device)], 
                                            dim=1)
        #print("noise {}".format(add_noise.shape))

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class Critic3d(nn.Module):
    def __init__(self, in_channels=3):
        super(Critic3d, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            #nn.ZeroPad3d((1, 0, 1, 0)),
            nn.Conv3d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

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
weights_init_normal(gen)
critic = Critic3d().to(device)
weights_init_normal(critic)

noise = torch.randn((1,100,16,16,8)).to(device) #Will be fed to generator's bottle neck
summary(gen, input_size=[(1,2, 256, 256, 128)]) # do a test pass through of an example input size 
summary(critic, input_size=[(8,1, 256, 256, 128),(8,2,256,256,128)])
