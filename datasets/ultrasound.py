import random
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, models
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn.functional import pad
from skimage.transform import resize
import time
import json
from data_transforms.ultrasound_transform imprt Ultrasound_Transform

class Ultrasound_Dataset(Dataset):
    def __init__(self, config, start=0, end=200, is_train=False, shuffle_list = True, apply_norm=True, no_text_mode=False, is_test=False):
        super().__init__()
        self.root_path = config['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.is_test = is_test
        self.label_names = config['label_names']
        self.num_classes = len(self.label_names)
        self.config = config
        self.apply_norm = apply_norm
        self.no_text_mode = no_text_mode
        label_dict = {
            'Liver': [100,0,100],
            'Kidney': [255,255,0],
            'Pancreas': [0,0,255],
            'Vessels': [255,0,0],
            'Adrenals': [0,255,255],
            'Gall Bladder': [0,255,0],
            'Bones': [255,255,255],
            'Spleen': [255,0,255]
        }

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

        #define data transform
        self.data_transform = Ultrasound_Transform(config=config)
    
    def populate_lists(self):
        if self.is_train:
            csv_file = os.path.join(self.root_path, 'train', 'train.csv')
        else:
            if self.is_test:
                csv_file = os.path.join(self.root_path, 'test', 'test.csv')
            else:
                csv_file = os.path.join(self.root_path, 'test', 'test.csv')
        
        df = pd.read_csv(csv_file)
        if self.is_train:
            root_path = os.path.join(self.root_path, 'train')
        else:
            root_path = os.path.join(self.root_path, 'test')

        for i in range(len(df)):
            self.img_path_list.append(os.path.join(root_path,df['img_path'][i]))
            self.img_names.append(df['img_path'][i])
            self.label_path_list.append(os.path.join(root_path,df['label_path'][i]))
            self.label_list.append(df['label_name'][i])
        

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['volume_channel']==2:
            img = img.permute(2,0,1)

        # if self.no_text_mode:
        #     label = torch.zeros((self.num_classes,img.shape[1],img.shape[2]))
        #     for i,label_name in enumerate(self.label_names):
        #         try:
        #             lbl_path = os.path.join(self.label_path_list[index],label_name.replace(' ','_')+'_labels',self.img_names[index])
        #             # print("lbl path: ", lbl_path)
        #             label_part = torch.Tensor(np.array(Image.open(lbl_path)))
        #         except:
        #             label_part = torch.zeros(img.shape[1], img.shape[2])
        #         label[i,:,:] = label_part
        #     label = (label>0)+0
            
        #     img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
        #     label = (label>=0.5)+0
        #     label_of_interest = ''
        #     # print("img shape: ",img.shape)
        #     # print("label shape: ", label.shape)
            
        # else:
        try:
            label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
        except:
            1/0
            label = torch.zeros(img.shape[1], img.shape[2])

        label = np.all(np.where(label==self.label_dict[self.label_list[index]],1,0),axis=2)
        
        label = torch.Tensor(label+0).unsqueeze(0)
        img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)

        #convert all grayscale pixels due to resizing back to 0, 1
        label = (label>=0.5)+0
        label = label[0]
        h,w = label.shape
        label_of_interest = self.label_list[index]


        return img, label, self.img_path_list[index], label_of_interest

    def __len__(self):
        return len(self.img_path_list)
