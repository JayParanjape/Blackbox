import os
import PIL.Image as Image
import sys
sys.path.append("..")
from modelling.model import FinalModel
import argparse
import yaml
from data_utils import get_data
import torch
import random
import numpy as np
from utils import *
import torchvision.transforms as T


label_names = ['Glands']
label_dict = {}
# visualize_dict = {}
for i,ln in enumerate(label_names):
        label_dict[ln] = i
        # visualize_dict[ln] = visualize_li[i]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config_tmp.yml',
                        help='config file path')

    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')

    parser.add_argument('--save_path', default='checkpoints/temp.pth',
                        help='pretrained model path')

    parser.add_argument('--device', default='cuda:0', help='device to train on')

    parser.add_argument('--labels_of_interest', default='Left Prograsp Forceps,Maryland Bipolar Forceps,Right Prograsp Forceps,Left Large Needle Driver,Right Large Needle Driver', help='labels of interest')

    args = parser.parse_args()

    return args

def main_predict(config, pretrained_path, save_path, device):

    #make folder to save visualizations
    os.makedirs(os.path.join(save_path,"rescaled_preds"),exist_ok=True)
    os.makedirs(os.path.join(save_path,"rescaled_gt"),exist_ok=True)
    
    encoder_config = config['encoder_config']
    decoder_config = config['decoder_config']
    prompt_encoder_config = config['prompt_encoder_config']
    blackbox_config = config['blackbox_config']
    optim_config = config['optimizer_config']
    data_config = config['data_config']
    train_config = config['train_config']

    dataset_dict, dataloader_dict, _ = get_data(data_config)
    transform = T.ToPILImage()

    model = FinalModel(encoder_config=encoder_config, decoder_config=decoder_config, blackbox_config=blackbox_config, prompt_config=prompt_encoder_config, device=device)
    model.data_pixel_mean = dataset_dict['test'].data_transform.pixel_mean
    model.data_pixel_std = dataset_dict['test'].data_transform.pixel_std

    model = model.to(device)
    if pretrained_path:
        model.decoder.load_state_dict(torch.load(pretrained_path,map_location=device), strict=True)
    print("debug: model loaded")

    dices = []
    for i in range(len(dataset_dict['test'])):
        image, label, _, text = dataset_dict['test'][i]
        image = image.unsqueeze(0).to(model.device)
        label = label.unsqueeze(0).to(model.device)
        points = []
        boxes = []
        texts = []

        if label.any() and not train_config['use_only_text']:
            _,y,x = torch.where(label==1)
            pos_prompts = torch.cat([x.unsqueeze(1),y.unsqueeze(1)],dim=1)
            pos_point_idx1 = random.randint(0,y.shape[0]-1)
            pos_point_idx2 = random.randint(0,y.shape[0]-1)
            random_positive_point = pos_prompts[pos_point_idx1]
            random_positive_point2 = pos_prompts[pos_point_idx2]
            if random_positive_point[0] < random_positive_point2[0]:
                random_bbox = torch.tensor([random_positive_point[0], random_positive_point[1], random_positive_point2[0], random_positive_point2[1]])
            else:
                random_bbox = torch.tensor([random_positive_point2[0], random_positive_point2[1], random_positive_point[0], random_positive_point[1]])
            point = random_positive_point
            box = random_bbox

            #choose only point
            if train_config['use_only_point']:
                points.append(point.unsqueeze(0).unsqueeze(0))
                boxes.append(None)
                texts.append(None)
            else:

                #choose one from text, point and bbox
                choice = np.random.choice([1,2,3])
                if choice==1:
                    box, text = None, None
                elif choice==2:
                    point, text = None, None
                else:
                    point, box = None
                points.append(point)
                boxes.append(box)
        else:
            points.append(None)
            boxes.append(None)
            texts.append(text)

        if train_config['use_only_point']:
            points = torch.cat(points,dim=0)
            print(point)
        else:
            print(texts)

        with torch.no_grad():
            output = model(image, points, boxes, texts)
            output = torch.Tensor(output).to(label.device)
        # print(torch.unique(output))
        dice = dice_coef(label, output)
        dices.append(dice)
        print(dice)

        #save image at path
        # print("Output shape: ", output.shape)
        # print("Label shape: ", label.shape)
        # print("Output nuique: ", torch.unique(output))
        pred = Image.fromarray((255*output.cpu()).numpy().astype(np.uint8).transpose(1, 2, 0)[:,:,0])
        mask = Image.fromarray((255*label.cpu()).numpy().astype(np.uint8).transpose(1, 2, 0)[:,:,0])

        pred.save(os.path.join(save_path, 'rescaled_preds', str(i)+'.png'), 'PNG')
        mask.save(os.path.join(save_path, 'rescaled_gt', str(i)+'.png'), 'PNG')
        # break

    print("Average dice scores: ", torch.mean(torch.Tensor(dices)))
    

if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # # for training the model
    main_predict(config, args.pretrained_path, args.save_path, device=args.device)