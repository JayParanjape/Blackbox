from modelling.model import FinalModel
from optimizers import *
import os
import sys
import numpy as np
import torch
import random


class Loss_fxn():
    def __init__(self, losses_list=[]):
        if losses_list==[]:
            self.losses_list = [torch.nn.CrossEntropyLoss()]
        else:
            self.losses_list = losses_list

    def forward(self, pred, label):
        loss = 0
        for l in self.losses_list:
            loss += l(pred, label)
        return loss

def train(dataset_dict, encoder_config, prompt_encoder_config, decoder_config, blackbox_config, optim_config, train_config, device, pretrained_path, save_path):
    # tr_dataloader, val_dataloader = iter(dataloader_dict['train']), iter(dataloader_dict['val'])
    tr_dataset, val_dataset = dataset_dict['train'], dataset_dict['val']
    # print("debug: len tr dataset", len(tr_dataset))

    num_training_iters = train_config['num_train']
    model = FinalModel(encoder_config=encoder_config, decoder_config=decoder_config, blackbox_config=blackbox_config, prompt_config=prompt_encoder_config, device=device)
    model.data_pixel_mean = tr_dataset.data_transform.pixel_mean
    model.data_pixel_std = tr_dataset.data_transform.pixel_std
    model = model.to(device)
    if pretrained_path:
        model.decoder.load_state_dict(torch.load(pretrained_path,map_location=device), strict=True)
    print("debug: model loaded")
    print(model.decoder)

    #define loss function
    losses_list = []
    if 'focal' in train_config['Loss']:
        losses_list.append(focal_loss)
    if 'dice' in train_config['Loss']:
        losses_list.append(dice_loss)
    loss_fxn = Loss_fxn(losses_list)

    print("debug: loss loaded")
    for i in range(1,1+num_training_iters):
        #TODO properly. get datapoint and add batch size

        image = []
        label = []
        points = []
        boxes = []
        text = []
        for j in range(train_config['batch_size']):
            data_idx = np.random.choice(len(tr_dataset))
            # image, point, box, text, label = tr_dataset[data_idx]
            image_j, label_j, _, text_j = tr_dataset[data_idx]
            image_j = image_j.unsqueeze(0).to(device)
            label_j = label_j.unsqueeze(0).to(device)
            image.append(image_j)
            label.append(label_j)
            # image, label, _, text = next(tr_dataloader)

            #get random positive point if it exists
            if label_j.any() and not train_config['use_only_text']:
                _,y,x = torch.where(label_j==1)
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
                    text.append(None)
                else:

                    #choose one from text, point and bbox
                    choice = np.random.choice([1,2,3])
                    if choice==1:
                        box, text_j = None, None
                    elif choice==2:
                        point, text_j = None, None
                    else:
                        point, box = None
                    points.append(point)
                    boxes.append(box)
                    text.append(text_j)
            else:
                points.append(None)
                boxes.append(None)
                text.append(text_j)


        image = torch.cat(image, dim=0)
        label = torch.cat(label, dim=0)
        if train_config['use_only_point']:
            points = torch.cat(points,dim=0)
        # print("debug: points shape ",points.shape)
        w = torch.nn.utils.parameters_to_vector(model.decoder.parameters())


        if optim_config['name']=='spsa-gc':
            with torch.no_grad():
                lr = optim_config['a']/((i + optim_config['o'])**optim_config['alpha'])
                ck = optim_config['c']/(i**optim_config['gamma'])
                ghat, loss, dice = spsa_grad_estimate_bi(model, image, points, boxes, text, label, loss_fxn, ck, optim_config['sp_avg'])
                if i==1:
                    m = ghat
                else:
                    m = optim_config['momentum']*m + ghat
                accum_ghat = ghat + optim_config['momentum']*m
                w = w - lr*accum_ghat
                torch.nn.utils.vector_to_parameters(w, model.decoder.parameters())

        #TODO printing loss and dice and saving model.
        print(f"Iteration: {i}, Loss: {loss}, dice: {dice}")

    return model