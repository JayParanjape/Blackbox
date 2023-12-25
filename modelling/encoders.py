import numpy as np
import torch
import torch.nn as nn
import clip

def get_image_embedding(img, encoder_config, device):
    with torch.no_grad:
        encoder = get_encoder(encoder_config, device)
        im_embeds = encoder.encode_image(img)
    return im_embeds

def get_encoder(encoder_config, device):
    if encoder_config['name'] == 'CLIP':
        return Clip_Encoder(device)
    return None


class Clip_Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.clip_model, _  = clip.load("ViT-B/32", device=device)

    def encode_image(self, img):
        return self.clip_model.encode_image(img)

class ViT_Encoder():
    pass

class Dino_Resnet50_Encoder():
    pass