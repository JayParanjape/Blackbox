import random
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_transforms.glas_transform import GLAS_Transform


class GLAS_Dataset(Dataset):
    def __init__(self, config, is_train=False, shuffle_list = True, apply_norm=True, no_text_mode=False, is_test=False) -> None:
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

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

        #define data transform
        self.data_transform = GLAS_Transform(config=config)

    def __len__(self):
        return len(self.img_path_list)

    def populate_lists(self):
        if self.is_train:
            imgs_path = os.path.join(self.root_path, 'train')
            labels_path = os.path.join(self.root_path, 'train')
        else:
            if self.is_test:
                imgs_path = os.path.join(self.root_path, 'test')
                labels_path = os.path.join(self.root_path, 'test')
            else:
                imgs_path = os.path.join(self.root_path, 'validation')
                labels_path = os.path.join(self.root_path, 'validation')


        for img in os.listdir(imgs_path):
            if (('jpg' not in img) and ('jpeg not in img') and ('png' not in img) and ('bmp' not in img)):
                continue
            if 'anno' in img:
                continue
            if self.no_text_mode:
                self.img_names.append(img)
                self.img_path_list.append(os.path.join(imgs_path,img))
                self.label_path_list.append(os.path.join(labels_path, img[:-4]+'_anno.bmp'))
                self.label_list.append('')
            else:
                for label_name in self.label_names:
                    self.img_names.append(img)
                    self.img_path_list.append(os.path.join(imgs_path,img))
                    self.label_path_list.append(os.path.join(labels_path, img[:-4]+'_anno.bmp'))
                    self.label_list.append(label_name)


    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['volume_channel']==2:
            img = img.permute(2,0,1)
            
        try:
            label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
            if len(label.shape)==3:
                label = label[:,:,0]
        except:
            label = torch.zeros(img.shape[1], img.shape[2])
        
        label = label.unsqueeze(0)
        label = (label>0)+0
        label_of_interest = self.label_list[index]

        #convert all grayscale pixels due to resizing back to 0, 1
        img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
        label = (label>=0.5)+0
        label = label[0]

        return img, label, self.img_path_list[index], label_of_interest
