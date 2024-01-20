from .decoder import get_decoder
from .encoders import get_encoder
from .prompt_encoder import PromptEncoder
from .blackbox import get_blackbox
import torch.nn as nn
import torch
import numpy as np

class FinalModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, blackbox_config, prompt_config, device):
        super().__init__()
        self.encoder = get_encoder(encoder_config, device)
        self.prompt_encoder = PromptEncoder(prompt_config, device=device)
        #to be set outside function
        self.data_pixel_mean = None
        self.data_pixel_std = None
        self.device = device
        self.use_sam_auto_mode = blackbox_config['auto_mode']
        self.use_sam_actual = blackbox_config['use_sam_actual']
        
        #set some decoder config params based on encoder and prompt encoder
        # if encoder_config['name']=='CLIP':
        decoder_config['prompt_input_dim'] = prompt_config['embedding_size']
        decoder_config['prompt_output_dim'] = decoder_config['prompt_output_dim']
        #total channels = 1078 = 21 X 49
        decoder_config['decoder_input_dim'] = decoder_config['decoder_input_dim']
        decoder_config['img_size'] = prompt_config['input_img_size']
        
        if decoder_config['name']=='SAM':
            if encoder_config['name']=='CLIP':
                decoder_config['encoder_dim'] = 512
            elif encoder_config['name']=='DINO-RESNET50':
                decoder_config['encoder_dim'] = 2048
                

        #TODO - figure out how to add positional embeddings. One way is through learnable embeddings        
        self.decoder = get_decoder(decoder_config=decoder_config, device=device)
        self.blackbox = get_blackbox(blackbox_config=blackbox_config, device=device)
        
        #encoder and blackbox have no gradients
        for i in self.encoder.parameters():
            i.requires_grad = False
        for i in self.blackbox.parameters():
            i.requires_grad = False
        for i in self.prompt_encoder.parameters():
            i.requires_grad = False

    def forward(self, img, point=None, box=None, text=None):
        prompt_embeddings = []
        use_sam_actual = self.use_sam_actual
        if text!=None and text[0]!= None:
            for i in range(len(text)):
                prompt_embeddings_i,_ = self.prompt_encoder(points = point[i], bboxes=box[i], text=text[i])
                prompt_embeddings.append(prompt_embeddings_i)
            prompt_embeddings = torch.cat(prompt_embeddings, dim=0)
        else:
            prompt_embeddings,_ = self.prompt_encoder(points = point, bboxes = box, text = text)

        img_embeddings = self.encoder.encode_image(img)
        # print("debug: img embeddings shape", img_embeddings.shape)
        # print("debug: prompt embeddings shape", prompt_embeddings.shape)
        
        if self.decoder.name=='concat':
            prompt_img = self.decoder(img_embeddings, prompt_embeddings)
        elif self.decoder.name=='sam_decoder':
            img_pe = self.prompt_encoder.pe_layer(img_embeddings.shape[-2:]).unsqueeze(0)
            prompt_img = self.decoder(img_embeddings, img_pe, prompt_embeddings, None)

        #add prompt image to image
        sam_img = img + prompt_img
        #convert image to uint8
        sam_img = (sam_img*self.data_pixel_std.unsqueeze(0).to(sam_img.device) + self.data_pixel_mean.unsqueeze(0).to(sam_img.device))
        sam_img = torch.clip(sam_img, 0, 255)

        #get output from black box model
        #point prompt api from sam only supports 1 image at a time.
        bs = img.shape[0]
        #TODO need cleaner condition
        if use_sam_actual:
            if len(sam_img.shape)==4:
                sam_img = sam_img[0]
                point = point[0]
            if sam_img.shape[0]==3:
                sam_img = sam_img.permute(1,2,0).cpu().numpy()
            sam_img = sam_img.astype(np.uint8)
            
        
        if self.use_sam_auto_mode:
            mask = self.blackbox(sam_img, None, None, None, bs, use_sam_actual=True)
        else:
            mask = self.blackbox(sam_img, point, box, text, bs, use_sam_actual=use_sam_actual)

        #this step required since labels hsa only 1 mask always
        if len(mask.shape)==4:
            mask = mask[:,0,:,:]

        return mask
