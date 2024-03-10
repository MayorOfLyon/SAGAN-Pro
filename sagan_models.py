import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np
from einops import rearrange
import random

random.seed(42)
torch.manual_seed(42)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        # add
        self.heads = 8
        self.scale = self.heads ** -0.5
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
        # self.gamma = nn.Parameter(torch.tensor(1.0))
        # self.beta = nn.Parameter(torch.tensor(1.0))
        self.softmax  = nn.Softmax(dim=-1) #
        
        dropout = 0.1
        # add(feed forward)
        self.net = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_dim, in_dim, kernel_size= 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            SpectralNorm(nn.Conv2d(in_dim, in_dim, kernel_size= 1)),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        # proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X C x (*W*H)
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height)# B X C x (*W*H)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        # add(multi-head)
        qkv = [proj_query, proj_key, proj_value]
        proj_query, proj_key, proj_value = map(lambda t: rearrange(t, 'b (h n) s -> b h s n', h = self.heads), qkv)
        # gai
        energy = torch.matmul(proj_query.transpose(-1, -2), proj_key) * self.scale
        attention = self.softmax(energy) # BX (N) X (N)
        out = torch.matmul(attention, proj_value.transpose(-1, -2))
        out = out.transpose(-1, -2) 
        out = rearrange(out, 'b h s n -> b (h n) s')
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        
        # add(feed forward)
        out_ = self.net(out)
        out = out + self.beta*out_
        
        return out,attention
    
class conv_block(nn.Module):
    def __init__(self, in_dim):
        super(conv_block, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(1)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv1 = SpectralNorm(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1))
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = SpectralNorm(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1))
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = SpectralNorm(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1))
        # self.gamma = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        out = self.relu1((self.conv1(self.pad1(x))))
        out = self.relu2((self.conv2(self.pad2(out))))
        out = self.conv3(self.pad3(out))
        out = self.gamma*out + x
        return out
        
class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        # attention layers
        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64, 'relu')
        
        # conv blocks
        self.conv_block1 = conv_block(128)
        self.conv_block2 = conv_block(64)
        
        self.conv_block3 = conv_block(512)
        self.conv_block4 = conv_block(256)
        
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.conv_block3(out)
        out = self.l2(out)
        out = self.conv_block4(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out = self.conv_block1(out)
        out = self.l4(out)
        out,p2 = self.attn2(out)
        out = self.conv_block2(out)
        out = self.last(out)
        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')
        
    def forward(self, x):
        out = self.l1(x) 
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2
