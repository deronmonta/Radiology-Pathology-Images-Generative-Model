import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
#Building blocks for 3D Unet

def maxpool3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool

def maxpool2d():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def avgpool3d(): 
    pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=1)
    return pool



def conv_block(in_dim, out_dim):
    layers = nn.Sequential(
        nn.Conv3d(in_dim,out_dim,kernel_size=4, stride=2,padding=1),
        nn.BatchNorm3d(out_dim), 
        nn.LeakyReLU(0.2,inplace=True)
    )
    return layers

def conv_block_2(in_dim, out_dim):
    
    layers = nn.Sequential(
        conv_block(in_dim,out_dim),
        nn.Conv3d(out_dim,out_dim, kernel_size=3,stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(0.2,inplace=True)
    )
    return layers

def conv_block_2d(in_dim, out_dim):
    layers = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=4, stride=2,padding=1),
        nn.BatchNorm2d(out_dim), 
        nn.LeakyReLU(0.2,inplace=True)
    )
    return layers

def conv_block_2_2d(in_dim, out_dim):
    
    layers = nn.Sequential(
        conv_block_2d(in_dim,out_dim),
        nn.Conv2d(out_dim,out_dim, kernel_size=3,stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2,inplace=True)
    )
    return layers

def dense(in_dim,out_dim):
    layer = nn.Sequential(
        nn.Linear(in_dim,out_dim)
    )
    return layer



def trans_conv_block(in_dim, out_dim):
    layers = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2,inplace=True)
    )
    return layers



def trans_conv_block_2(in_dim, out_dim):
    layers = nn.Sequential(
        trans_conv_block(in_dim,in_dim),
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4,stride=2,padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2,inplace=True)
    )
    return layers

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_block(in_channels, out_channels)
        self.conv2 = conv_block(out_channels, out_channels)
        
        self. relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out  = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out