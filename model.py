import torch
import torch.nn as nn


class Critic3d(nn.Module):
    """Creates the convolutional network
    
    Replicates critic network from https://arxiv.org/abs/2202.07077
    consisting of six consecutive transposed 3D convolutions.

    Returns:
        A float as continuous and quantitative rating"""
    
    def __init__(self):
        super(Critic3d, self).__init__()
        self.c1 = self._block(3,128)
        self.c2 = self._block(128,128)
        self.c3 = self._block(128,128)
        self.c4 = self._block(128,128)
        self.c5 = self._block(128,128)
        self.c6 = self._block(128,1) # nn.Conv3d(128, 1, 3, 2, 1, bias=False)


    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels,kernel_size=3, stride =2, padding =1, bias=False),
            nn.SiLU(),
            nn.Dropout3d(p=0.15)
        )

    def forward(self, x, conditional_input):
        cat = torch.cat([x, conditional_input], dim=1)
        x = self.c1(cat)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        # Flatten the tensor
        flattened = torch.flatten(x, start_dim=2)
        # Reshape the tensor to [N, 1]
        out = flattened.view(conditional_input.shape[0] , -1) # conditional_input.shape[0] number of data in that batch
        linear = nn.Linear(out.shape[1],1)     
        return linear(out)



class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

        

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool3d((2,2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        #print(f"x up concat with skip {x.shape, skip.shape}")
        x = torch.cat([x, skip], axis=1)
        #print(f"after concat {x.shape}")
        x = self.conv(x)
        #print(f"after conv {x.shape}")
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
      
        """ Encoder """
        self.e1 = encoder_block(2, 64) #2 input channel because concatenated inputs
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = conv_block(612, 1024) #add 100 dimension for noise 
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d4 = decoder_block(128, 64)
        """ Classifier """
        self.outputs = nn.Conv3d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs, noise):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        #print("s1 ,p1 {}, {}".format(s1.shape, p1.shape))
        s2, p2 = self.e2(p1)
        #print("s2 ,p2 {}, {}".format(s2.shape, p2.shape)
        s3, p3 = self.e3(p2)
        #print("s3 ,p3 {}, {}".format(s3.shape, p3.shape))   
        s4, p4 = self.e4(p3)
        #print("s4 ,p4 {}, {}".format(s4.shape, p4.shape)) 
        """ Bottleneck """
        # Concatenate input tensor and noise tensor along the channel dimension (dim=1)
        p4_noise = torch.cat([p4, noise], dim=1)
        b = self.b(p4_noise)
        #print("b {}".format(b.shape))
        """ Decoder """
        d1 = self.d1(b, s4)
        #print("d1 {}".format(d1.shape))
        d2 = self.d2(d1, s3)
        #print("d2 {}".format(d2.shape))
        d3 = self.d3(d2, s2)
        #print("d3 {}".format(d3.shape))
        d4 = self.d4(d3, s1)
        #print("d4 {}".format(d4.shape))        
        """ Classifier """       
        return self.outputs(d4)



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
    BATCH_SIZE,C,H,W,D = real.shape
    epsilon= torch.rand((BATCH_SIZE, 1,1,1,1)).repeat(1, C, H, W,D).to(device) # one value for each sample and repeat(expand the dimensions) for desired C,H,W
    interpolated_images = real*epsilon + fake*(1-epsilon)
    # calculate critic scores
    mixed_scores = critic(interpolated_images, cond)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        )[0] # get first element

    #get only number of samples gradient.shape[0] and flatten other dimensions
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2,dim=1) #dimension=1
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty