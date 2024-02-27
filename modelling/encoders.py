import numpy as np
import torch
import torch.nn as nn
import clip
from transformers import ResNetModel, ViTForImageClassification

def get_image_embedding(img, encoder_config, device):
    with torch.no_grad:
        encoder = get_encoder(encoder_config, device)
        im_embeds = encoder.encode_image(img)
    return im_embeds

def get_encoder(encoder_config, device):
    if encoder_config['name'] == 'CLIP':
        return Clip_Encoder(device)
    elif encoder_config['name'] == 'DINO-RESNET50':
        return Dino_Resnet50_Encoder(device)
    elif encoder_config['name'] == 'VIT-MAE-BASE':
        return ViT_MAE_Encoder(device)
    elif encoder_config['name'] == 'VIT':
        return ViT(device)
    return None


class Clip_Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.clip_model, _  = clip.load("ViT-B/32", device=device)

    def encode_image(self, img, perform_pool=False):
        # self.clip_model.image_resolution = img.shape[-1] if img.shape[-1]!=3 else img.shape[1]
        return self.clip_model.encode_image(img)

class ViT_MAE_Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = ViTForImageClassification.from_pretrained("facebook/vit-mae-base").to(device)
        self.embed_dim = 768
    
    def encode_image(self, img, perform_pool=True):
        out = self.backbone(img, output_hidden_states=True, interpolate_pos_encoding=True)
        z = out.hidden_states[-1][:,0,:] #NX768 by using the CLS from last hidden state
        return z

class ViT(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
        self.embed_dim = 768
    
    def encode_image(self, img, perform_pool=True):
        out = self.backbone(img, output_hidden_states=True, interpolate_pos_encoding=True)
        z = out.hidden_states[-1][:,0,:] #NX768 by using the CLS from last hidden state
        return z

class Dino_Resnet50_Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = ResNetModel.from_pretrained("Ramos-Ramos/dino-resnet-50").to(device)
        self.embed_dim = 2048
    
    def encode_image(self, img, perform_pool=True):
        out = self.backbone(img).last_hidden_state
        b,c,h,w = out.shape
        #out shape B X 2048 X 7 X 7
        if perform_pool:
            out = nn.functional.adaptive_avg_pool2d(out,(1,1)).view(b,c)
        return out
