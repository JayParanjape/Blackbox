import torch
import torch.nn as nn
from segment_anything import build_sam, sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import clip

def get_blackbox(blackbox_config, device):
    if blackbox_config['name']=='SAM':
        return BBox_SAM(blackbox_config, device)

class BBox_SAM(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.sam_model = sam_model_registry[config['model_type']](checkpoint=config['sam_checkpoint'])
        self.sam_model = self.sam_model.to(device=device)
        self.clip_model,_ = clip.load("ViT-B/32", device=device)
        self.automatic_mask_generator = SamAutomaticMaskGenerator(self.sam_model)
        self.prompt_mask_generator = SamPredictor(self.sam_model)
        self.transform = self.prompt_mask_generator.transform

    def forward(self, img, point=None, box=None, text=None):
        with torch.no_grad():
            img_size = img.shape[-1]
            if point==None and box==None and text==None:
                #automatic case
                masks = self.automatic_mask_generator.generate(img)
            else:
                if text:
                    #support for text not provided in SAM API. Simulated here
                    #resize and transform image
                    # print("debug: input img to image encoder: ", img.shape)
                    img = self.transform.apply_image_torch(img)
                    img = self.sam_model.preprocess(img)
                    # print("debug: after prerocessing by blackbox: ", img.shape)

                    image_embeddings = self.sam_model.image_encoder(img)
                    text_inputs = (clip.tokenize(text)).to(self.device)
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = text_features.unsqueeze(1)[:,:,:256]

                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    sparse_embeddings = sparse_embeddings.repeat(text_features.shape[0],1,1)

                    sparse_embeddings = torch.cat([sparse_embeddings, text_features], dim=1)
                    # print("debug: sparse embeddings ", sparse_embeddings.shape)
                    # print("debug: image embeddings ", image_embeddings.shape)

                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    # print("debug: low res masks shape: ", low_res_masks.shape)
                    masks = self.sam_model.postprocess_masks(low_res_masks, (img_size,img_size), (img_size,img_size))
                    # print("debug: masks shape: ", masks.shape)


                else:
                    self.prompt_mask_generator.set_image(img)
                    #only supports positive points
                    if point:
                        points_labels = torch.ones(point.shape[0]).to(self.device)
                    masks, _,_ = self.prompt_mask_generator.predict(point_coords=point, point_labels=points_labels, box=box)

            return masks