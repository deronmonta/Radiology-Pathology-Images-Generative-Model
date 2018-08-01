

import torch
import torch
import torch.nn as nn
from nets.blocks import *
import torch.nn.functional as F


class Encoder(nn.Module):
    '''
    Encode a 3D MR image to a single vector latent distribution
    '''
    def __init__(self,in_dim,out_dim,num_filters):
        super(Encoder,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        self.conv1 = conv_block_2(self.in_dim,num_filters*1)
        self.maxpool1 = maxpool3d()
        self.conv2 = conv_block_2(num_filters*1,num_filters*2)
        self.maxpool2 = maxpool3d()
        self.conv3 = conv_block_2(num_filters*2,num_filters*4)
        self.maxpool3 = maxpool3d()
        self.conv4 = conv_block_2(num_filters*4,num_filters*8)
        self.fully_connected = dense(512,self.out_dim)

    def forward(self,input):
        x = self.conv1(input)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = x.view(-1,512)
        x = self.fully_connected(x)
        return x

class Generator(nn.Module):
    '''
    Decodes a encoded latent representation to a artificial 2D histopathology image 
    '''
    def __init__(self,in_dim,out_dim,num_filters):
        super(Generator,self).__init__()
        self.up1 = trans_conv_block_2(in_dim,num_filters*8)
        self.up2 = trans_conv_block_2(num_filters*8,num_filters*4)
        self.up3 = trans_conv_block_2(num_filters*4,num_filters*2)
        self.up4 = trans_conv_block_2(num_filters*2,num_filters*1)
        self.up5 = trans_conv_block(num_filters*1,out_dim)
        self.up6 = trans_conv_block(out_dim,out_dim)

    def forward(self,input):
        x = self.up1(input)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        x = F.tanh(x)
        #print('G output shape {}'.format(x.shape))
        print(x.shape)
        return x

class Discriminator(nn.Module):
    '''
    This is the discriminator for artificial image and true image
    '''
    def __init__(self,in_dim,out_dim,num_filters):
        
        super(Discriminator,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        self.conv1 = conv_block_2_2d(self.in_dim, self.num_filters*1)
        self.conv2 =conv_block_2_2d(self.num_filters*1,self.num_filters*2)
        self.conv3 = conv_block_2_2d(self.num_filters*2,self.num_filters*4)
        
        self.maxpool1 = maxpool2d()
        self.maxpool2 = maxpool2d()
        self.maxpool3 = maxpool2d()
        
        self.fc = dense(16384,self.out_dim)
        

    def forward(self,input):
        x = self.conv1(input)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        print(x.shape)
        x = x.view(-1,16384)
        x = self.fc(x)
        x = F.sigmoid(x)

        return x