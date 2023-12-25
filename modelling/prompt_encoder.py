import numpy as np
import torch
import torch.nn as nn
import clip

from typing import Any, Optional, Tuple, Type

#adapted in part from SAM
class PromptEncoder(nn.Module):
    def __init__(self, config, device='cuda:0') -> None:
        super().__init__()
        self.embedding_size = config['embedding_size']//2
        self.imput_img_size = config['input_img_size']
        self.device=device

        #for point based and bounding box based prompts
        self.pe_layer = PositionEmbeddingRandom(self.embedding_size)
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        # point_embeddings = [nn.Embedding(1, self.embedding_size) for i in range(self.num_point_embeddings)]
        # self.point_embeddings = nn.ModuleList(point_embeddings)
        # self.not_a_point_embed = nn.Embedding(1, self.embedding_size)

        #for text based prompt
        self.clip_model, _  = clip.load("ViT-B/32", device=device)
        self.text_affine_layer = nn.Sequential(
            nn.Linear(512, self.embedding_size),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_size)
        )

    def encode_point(self, points):
        points = points + 0.5 #move to the pixel center from the top left point
        points_embedding = self.pe_layer.forward_with_coords(points, self.input_img_size)
        #only consider positive points for now
        # points_embedding += self.point_embeddings[1].weight
        return points_embedding  #N1 X 1 X embedding_size

    def encode_bb(self, boxes):
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_img_size)
        # corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        # corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding #N2 X 2 X embedding_size

    def encode_mask(self, masks):
        #will add functionality later
        pass

    def encode_text(self, text):
        text_inputs = (clip.tokenize(text)).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs)
            text_embedding = text_embedding.unsqueeze(1)
            # text_embedding = self.text_affine_layer(text_embedding)
            return text_embedding

    def forward(self, points=None, bboxes=None, masks=None, text=None):
        sparse_prompt_embeddings = []
        dense_prompt_embeddings = []
        #encode points
        if points:
            point_prompt = self.encode_point(points)
            sparse_prompt_embeddings.append(point_prompt)

        #encode bounding boxes
        if bboxes:
            bbox_prompt = self.encode_bb(bboxes)
            sparse_prompt_embeddings.append(bbox_prompt)

        #encode mask
        if masks:
            mask_prompt = self.encode_mask(masks)
            dense_prompt_embeddings.append(mask_prompt)

        #encode text
        if text:
            text_prompt = self.encode_text(text)
            # print("debug: , encode text shape: ", text_prompt.shape)
            sparse_prompt_embeddings.append(text_prompt)

        sparse_prompt_embeddings = torch.cat(sparse_prompt_embeddings, dim=0)
        # print("debug: sparse prompt embeddings shape: ", sparse_prompt_embeddings.shape)
        if not masks:
            dense_prompt_embeddings = None

        return sparse_prompt_embeddings, dense_prompt_embeddings


#class for initializing positional embeddings
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
