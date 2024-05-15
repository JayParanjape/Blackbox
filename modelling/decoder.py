import torch 
import torch.nn as nn
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
        src = src.view(b, -1, h, w)
        b, c, h, w = src.shape

        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        # Run the transformer
        hs, src = self.transformer(src.float(), pos_src, tokens.float())

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, -1, h, w)
        out = self.output_upscaling(src)
        b,c,h,w = out.shape

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
        self.final_img_size = decoder_config['img_size']
        self.auto_sam = decoder_config['auto_mode'] 
        
        #for auto case: 
        if self.auto_sam:
            try:
                unit = self.final_img_size//32
                self.trainable_parameter = nn.Parameter(torch.zeros((self.dec_input-self.prompt_input_dim, unit, unit)))
            except:
                pass
        # else:
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
        #img embeds: B X C or BXCXdXd
        #prompt_embeds: B X d1 X d2
        if len(img_embeds.shape)==4:
            b,c,d1,d2 = img_embeds.shape
            concat_img = torch.cat([img_embeds, self.trainable_parameter.repeat(b,1,1,1)], dim=1)
            prompt_img = self.decoder(concat_img)
        else:
            b,c = img_embeds.shape
            b2, d1,d2 = prompt_embeds.shape
            assert(b==b2)
            img_embeds = img_embeds.repeat(d1,1)
            prompt_embeds = prompt_embeds.view(b2*d1,d2).float()
            prompt_embeds = self.Prompt_Embedding_Converter(prompt_embeds)
            concat_img = torch.cat([img_embeds, prompt_embeds], dim=-1)
            
            #concat img dim: Bd1 X (multiple of (img_size/32)**2)
            unit = self.final_img_size//32
            concat_img = concat_img.view(b2*d1,-1,unit,unit)

            prompt_img = self.decoder(concat_img)
            #average across all prompts
            prompt_img = prompt_img.view(b2,d1,3,self.final_img_size,self.final_img_size)
            prompt_img = torch.mean(prompt_img, dim=1)
        
        return prompt_img

