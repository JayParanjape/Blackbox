import random
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from data_transforms.chestxdet_transform import ChestXDet_Transform

class ChestXDet_Dataset(Dataset):
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
            'Effusion': 1, 
            'Nodule': 2, 
            'Cardiomegaly': 3, 
            'Fibrosis': 4, 
            'Consolidation': 5, 
            'Emphysema': 6, 
            'Mass': 7, 
            'Fracture': 8, 
            'Calcification': 9, 
            'Pleural Thickening': 10, 
            'Pneumothorax': 11, 
            'Atelectasis': 12, 
            'Diffuse Nodule': 13
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
        self.data_transform = ChestXDet_Transform(config=config)
    
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

        try:
            label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
        except:
            1/0
            label = torch.zeros(img.shape[1], img.shape[2])

        if len(label.shape)==3:
            label = label[:,:,0]
        
        label = (label==self.label_dict[self.label_list[index]])
            
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
