import torch
from glob import glob
import os
import pandas as pd
import torch.utils.data as data
from utils.helpers import *

class Radiology_Pathology_Dataset(data.Dataset):
    def __init__(self,img_dir):
        '''
        img_dir: path to image directory.
       
        '''
        self.img_dir= img_dir
        self.dataframe = create_dataframe(self.img_dir)
        print(len(os.listdir(self.img_dir)))
        print(self.dataframe)
        

       

    def __getitem__(self,index):
        T1_img = load_volume(self.dataframe.iloc[index,1])# 2nd column for t1 image
        T2_img = load_volume(self.dataframe.iloc[index,2])# 3rd column for t2 image
        T1C_img = load_volume(self.dataframe.iloc[index,3])# 4th column for t1C image
        pathology_img = load_image(self.dataframe.iloc[index,5])#5th column for pathology
        sample = {'T1':T1_img,'T2':T2_img,'T1C':T1C_img,'pathology':pathology_img}

        return sample


    def __len__(self):
        
        return len(os.listdir(self.img_dir))