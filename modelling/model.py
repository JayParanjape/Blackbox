from .decoder import get_decoder
from .encoders import get_encoder
from .prompt_encoder import PromptEncoder
from .blackbox import get_blackbox
import torch.nn as nn
import torch


class FinalModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, blackbox_config, prompt_config, device):
        super().__init__()
        self.encoder = get_encoder(encoder_config, device)
        self.prompt_encoder = PromptEncoder(prompt_config, device=device)
        
        #set some decoder config params based on encoder and prompt encoder
        if encoder_config['name']=='CLIP':
            decoder_config['prompt_input_dim'] = 512
            decoder_config['prompt_output_dim'] = 517
            #total channels = 1078 = 21 X 49
            decoder_config['decoder_input_dim'] = 21

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
        for i in range(len(text)):
            prompt_embeddings_i,_ = self.prompt_encoder(points = point[i], bboxes=box[i], text=text[i])
            prompt_embeddings.append(prompt_embeddings_i)
        prompt_embeddings = torch.cat(prompt_embeddings, dim=0)

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

        #get output from black box model
        mask = self.blackbox(sam_img, point, box, text)

        #this step required since labels hsa only 1 mask always
        if len(mask.shape)==4:
            mask = mask[:,0,:,:]

        return mask