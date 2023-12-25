import torch 
import torch.nn as nn
import numpy as np
from common import TwoWayTransformer, LayerNorm2d



def get_decoder(decoder_config, device):
    if decoder_config['name']=='SAM':
        return SAM_Decoder(decoder_config['transformer_dim'])
    elif decoder_config['name']=='Concat':
        return Concat_Decoder(decoder_config)

class SAM_Decoder(nn.Module):
    def __init__(self, transformer_dim, device):
        super().__init__()
        self.name = 'sam_decoder'
        self.transformer_dim = transformer_dim
        self.device = device
        self.trasformer = TwoWayTransformer().to(device)
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                    LayerNorm2d(transformer_dim // 4),
                    nn.GELU(),
                    nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                    nn.GELU(),
                ).to(device)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor = None,
    ):
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens        
        tokens = sparse_prompt_embeddings

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings

        # src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        # print("image_pe.shape: ", image_pe.shape)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        mask_tokens_out = hs[:, 1, :]
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        out = self.output_upscaling(src)
        return out        


class Concat_Decoder(nn.Module):
    def __init__(self, decoder_config, device='cpu'):
        super().__init__()
        self.device=device
        self.name='concat'
        self.prompt_input_dim = decoder_config['prompt_input_dim']
        self.dec_input = decoder_config['decoder_input_dim']
        self.prompt_output_dim = decoder_config['prompt_output_dim']
        self.Prompt_Embedding_Converter = nn.Sequential(
            nn.Linear(self.prompt_input_dim, self.prompt_output_dim),
            nn.LayerNorm(self.prompt_output_dim),
            nn.GELU()
        ).to(device)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.dec_input, 64, 2,2,0),
            nn.ConvTranspose2d(64,64,1,1,0),
            LayerNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 64, 2,2,0),
            nn.ConvTranspose2d(64,32,1,1,0),
            LayerNorm2d(32),
            nn.GELU(),

            nn.ConvTranspose2d(32, 32, 2,2,0),
            nn.ConvTranspose2d(32,32,1,1,0),
            LayerNorm2d(32),
            nn.GELU(),

            nn.ConvTranspose2d(32, 32, 2,2,0),
            nn.ConvTranspose2d(32,16,1,1,0),
            LayerNorm2d(16),
            nn.GELU(),

            nn.ConvTranspose2d(16,3,2,2,0)
        ).to(device)

    def forward(self, img_embeds, prompt_embeds):
        #img embeds: B X C
        #prompt_embeds: B X d1 X d2
        b,c = img_embeds.shape
        b2, d1,d2 = prompt_embeds.shape
        assert(b==b2)
        img_embeds = img_embeds.repeat(d1,1)
        prompt_embeds = prompt_embeds.view(b2*d1,d2).float()
        # print("debug: prompt embeds dtype ", prompt_embeds.dtype)
        prompt_embeds = self.Prompt_Embedding_Converter(prompt_embeds)
        concat_img = torch.cat([img_embeds, prompt_embeds], dim=-1)
        #concat img dim: Bd1 X (multiple of 49)
        concat_img = concat_img.view(b2*d1,-1,7,7)

        prompt_img = self.decoder(concat_img)
        # print("debug: prompt img shape: ", prompt_img.shape)
        #average across all prompts
        prompt_img = prompt_img.view(b2,d1,3,224,224)
        prompt_img = torch.mean(prompt_img, dim=1)
        return prompt_img

