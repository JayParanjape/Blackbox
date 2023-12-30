import torch 
import torch.nn as nn
import numpy as np
from common import TwoWayTransformer, LayerNorm2d, MLP



def get_decoder(decoder_config, device):
    if decoder_config['name']=='SAM':
        return SAM_Decoder(decoder_config, device)
    elif decoder_config['name']=='Concat':
        return Concat_Decoder(decoder_config)

class SAM_Decoder(nn.Module):
    def __init__(self, decoder_config, device):
        super().__init__()
        self.name = 'sam_decoder'
        self.transformer_dim = decoder_config['transformer_dim']
        self.device = device
        self.img_size = decoder_config['img_size']

        #convert any dim img embedding to BXprompt_embed_dimXHXW
        self.embed_converter = MLP(decoder_config['encoder_dim'], decoder_config['prompt_embed_dim'], decoder_config['prompt_embed_dim'], 1)
        
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=decoder_config['prompt_embed_dim'],
            mlp_dim=2048,
            num_heads=8,
        ).to(device)
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
                    LayerNorm2d(self.transformer_dim // 4),
                    nn.GELU(),
                    nn.ConvTranspose2d(self.transformer_dim // 4, self.transformer_dim // 4, kernel_size=2, stride=2),
                    LayerNorm2d(self.transformer_dim // 4),
                    nn.GELU(),
                    nn.ConvTranspose2d(self.transformer_dim//4, self.transformer_dim // 4, kernel_size=2, stride=2),
                    LayerNorm2d(self.transformer_dim // 4),
                    nn.GELU(),
                    nn.ConvTranspose2d(self.transformer_dim//4, self.transformer_dim // 8, kernel_size=2, stride=2),
                    LayerNorm2d(self.transformer_dim // 8),
                    nn.GELU(),
                    nn.ConvTranspose2d(self.transformer_dim//8, 3, kernel_size=2, stride=2),
                    # nn.GELU(),
                ).to(device)

        # self.hypernetwork = MLP(decoder_config['prompt_embed_dim'], self.transformer_dim, self.transformer_dim//8, 3).to(device)
        print("initialized decoder")

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
        b, c, h, w = src.shape
        #convert image embeddings to prompt embed shape
        src = self.embed_converter(src.view(b,h,w,c))
        # print("debug: after embed converter src shape ", src.shape)
        src = src.view(b, -1, h, w)
        b, c, h, w = src.shape

        # src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        # print("image_pe.shape: ", image_pe.shape)
        # print("debug: src shape ", src.shape, src.dtype)
        # print("debug: pos src shape: ", pos_src.shape)
        # print("debug: tokens shape: ", tokens.shape, tokens.dtype)

        # Run the transformer
        hs, src = self.transformer(src.float(), pos_src, tokens.float())
        # print(f"debug: hs shape {hs.shape} src shape {src.shape}")

        # mask_tokens_out = hs[:, 0, :]
        # print(f"debug mask tokens shape {mask_tokens_out.shape}")
        # mask_tokens_out = self.hypernetwork(mask_tokens_out)
        # print(f"debug mask tokens after hypernet shape {mask_tokens_out.shape}")

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, -1, h, w)
        out = self.output_upscaling(src)
        b,c,h,w = out.shape
        # print(f"debug out shape {out.shape}")
        # out = (mask_tokens_out @ out.view(b, c, h * w)).view(b, c, h, w)

        #interpolate back to img size
        out = nn.functional.interpolate(
                out,
                (self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
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

