import torch.nn as nn
import torch
import numpy as np
from .blackbox import get_blackbox

class Baseline_VPT(nn.Module):
    def __init__(self, encoder_config, decoder_config, blackbox_config, prompt_config, device):
        super().__init__()
        img_size = prompt_config['input_img_size']
        self.vp = nn.Parameter(torch.zeros(3, img_size, img_size),requires_grad=False).to(device)
        self.blackbox = get_blackbox(blackbox_config=blackbox_config, device=device)
        self.data_pixel_std = None
        self.data_pixel_std = None
        self.use_sam_actual = True
        self.use_sam_auto_mode = False
        self.device = device

    def forward(self, img, point=None, box=None, text=None, return_sam_img = False, debug=False):
        sam_img = img + self.vp.unsqueeze(0)
        #convert image to uint8
        sam_img = (sam_img*self.data_pixel_std.unsqueeze(0).to(sam_img.device) + self.data_pixel_mean.unsqueeze(0).to(sam_img.device))
        sam_img = torch.clip(sam_img, 0, 255)

        diff_img = (self.vp*self.data_pixel_std.unsqueeze(0).to(sam_img.device) + self.data_pixel_mean.unsqueeze(0).to(sam_img.device))
        diff_img = torch.clip(diff_img, 0, 255)
        

        diff_img = diff_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)


        #get output from black box model
        #point prompt api from sam only supports 1 image at a time.
        bs = img.shape[0]
        #TODO need cleaner condition
        if len(sam_img.shape)==4:
            sam_img = sam_img[0]
            try:
                point = point[0]
            except:
                pass

        if sam_img.shape[0]==3:
            sam_img = sam_img.permute(1,2,0).cpu().numpy()
        sam_img = sam_img.astype(np.uint8)
        
        mask = self.blackbox(sam_img, point, box, text, bs, use_sam_actual=True)

        #this step required since labels hsa only 1 mask always
        if len(mask.shape)==4:
            mask = mask[:,0,:,:]

        if return_sam_img:
            return mask, sam_img, diff_img

        return mask

