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
from data_transforms.endovis_transform import ENDOVIS_Transform

class Endovis_Dataset(Dataset):
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
        self.label_dict = {
            'Left Prograsp Forceps':2,
            'Maryland Bipolar Forceps': 1,
            'Right Prograsp Forceps': 2,
            'Left Large Needle Driver':3,
            'Right Large Needle Driver':3,
            'Left Grasping Retractor':5,
            'Right Grasping Retractor':5,
            'Vessel Sealer':4,
            'Monopolar Curved Scissors':6
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
        self.data_transform = ENDOVIS_Transform(config=config)
    
    def populate_lists(self):
        if self.is_train:
            csv_file = os.path.join(self.root_path, 'train.csv')
        else:
            if self.is_test:
                csv_file = os.path.join(self.root_path, 'test.csv')
            else:
                csv_file = os.path.join(self.root_path, 'val.csv')
        df = pd.read_csv(csv_file)
        root_path = "/media/ubuntu/New Volume/jay/endovis17/"
        for i in range(len(df)):
            self.img_path_list.append(os.path.join(root_path,df['img_path'][i]))
            self.img_names.append(df['img_name'][i])
            self.label_path_list.append(os.path.join(root_path,df['label_path'][i]))
            self.label_list.append(df['label_name'][i])
        

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['volume_channel']==2:
            img = img.permute(2,0,1)

        if self.no_text_mode:
            label = torch.zeros((self.num_classes,img.shape[1],img.shape[2]))
            for i,label_name in enumerate(self.label_names):
                try:
                    lbl_path = os.path.join(self.label_path_list[index],label_name.replace(' ','_')+'_labels',self.img_names[index])
                    # print("lbl path: ", lbl_path)
                    label_part = torch.Tensor(np.array(Image.open(lbl_path)))
                except:
                    label_part = torch.zeros(img.shape[1], img.shape[2])
                label[i,:,:] = label_part
            label = (label>0)+0
            
            img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
            label = (label>=0.5)+0
            label_of_interest = ''
            # print("img shape: ",img.shape)
            # print("label shape: ", label.shape)
            
        else:
            try:
                label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
            except:
                1/0
                label = torch.zeros(img.shape[1], img.shape[2])

            
            label = label.unsqueeze(0)
            if self.is_test:
                label = (label==self.label_dict[self.label_list[index]]) + 0
            label = (label>0)+0
            label_of_interest = self.label_list[index]
            img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)

            #convert all grayscale pixels due to resizing back to 0, 1
            label = (label>=0.5)+0
            label = label[0]
            h,w = label.shape

            if self.is_test:
                label_name = self.label_list[index]
                if 'Left' in label_name:
                    side_mask = np.concatenate([label[:,:w//2],label[:,w//2:]*0], axis=1)
                elif 'Right' in label_name:
                    side_mask = np.concatenate([label[:,:w//2]*0, label[:,w//2:]], axis=1)
                else:
                    side_mask = label
            else:
                side_mask = label
            side_mask = torch.Tensor(side_mask).to(img.device)

        return img, side_mask, self.img_names[index], label_of_interest

    def __len__(self):
        return len(self.img_path_list)
