import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio, strategy: str = 'random', grid_size: int = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.strategy = strategy
        self.grid_size = grid_size  # number of patches per side (e.g., 16 for 32/2)

    def _grid_coords(self, T):
        H = W = self.grid_size if self.grid_size is not None else int(np.sqrt(T))
        return H, W

    def _make_order_for_sample(self, T):
        """Return a permutation array with visible indexes first, then masked indexes."""
        vis_T = int(T * (1 - self.ratio))
        if self.strategy == 'random' or self.grid_size is None:
            order = np.random.permutation(T)
            return order, vis_T

        H, W = self._grid_coords(T)
        # map linear index to (i,j)
        coords = np.arange(T).reshape(H, W)

        if self.strategy == 'grid':
            # keep 1/stride^2 visible; for 75% mask, stride=2 keeps 25% visible
            stride = 2
            keep = coords[::stride, ::stride].reshape(-1)
            keep = keep[:vis_T]  # in case rounding differs
            mask = np.setdiff1d(np.arange(T), keep, assume_unique=False)
            order = np.concatenate([np.random.permutation(keep), np.random.permutation(mask)])
            return order, vis_T

        if self.strategy == 'block':
            # mask a contiguous square block covering ~ ratio fraction
            mask_area = int(round(T * self.ratio))
            side = max(1, min(H, int(np.sqrt(mask_area))))
            # pick random top-left for masked block
            i0 = np.random.randint(0, H - side + 1)
            j0 = np.random.randint(0, W - side + 1)
            masked = coords[i0:i0+side, j0:j0+side].reshape(-1)
            masked = np.unique(masked)
            keep = np.setdiff1d(np.arange(T), masked, assume_unique=False)
            # ensure exact visible count if possible
            if keep.size > vis_T:
                keep = np.random.permutation(keep)[:vis_T]
            order = np.concatenate([np.random.permutation(keep), np.random.permutation(np.setdiff1d(np.arange(T), keep))])
            return order, vis_T

        # fallback to random
        order = np.random.permutation(T)
        return order, vis_T

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        orders = []
        vis_T = int(T * (1 - self.ratio))
        for _ in range(B):
            order, vis_count = self._make_order_for_sample(T)
            orders.append(order)
            vis_T = vis_count  # same for all samples
        forward_indexes = torch.as_tensor(np.stack(orders, axis=-1), dtype=torch.long, device=patches.device)
        backward_indexes = torch.argsort(forward_indexes, dim=0)
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:vis_T]
        return patches, forward_indexes, backward_indexes, vis_T

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 mask_sampling: str = 'random',
                 encoder_with_mask_token: bool = False,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio, strategy=mask_sampling, grid_size=image_size // patch_size)
        self.encoder_with_mask_token = encoder_with_mask_token
        if self.encoder_with_mask_token:
            self.mask_token_enc = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        T, B, C = patches.shape

        if self.encoder_with_mask_token:
            # shuffle without pos embedding, then fill masked positions with encoder mask tokens, reorder to original, then add pos embedding
            vis_patches, forward_indexes, backward_indexes, vis_T = self.shuffle(patches)
            masked_T = T - vis_T
            if masked_T > 0:
                mask_tokens = self.mask_token_enc.expand(masked_T, B, -1)
                seq = torch.cat([vis_patches, mask_tokens], dim=0)  # still in shuffled order
            else:
                seq = vis_patches
            seq = take_indexes(seq, backward_indexes)  # restore original order
            seq = seq + self.pos_embedding  # add pos embedding
            seq = torch.cat([self.cls_token.expand(-1, B, -1), seq], dim=0)
            seq = rearrange(seq, 't b c -> b t c')
            features = self.layer_norm(self.transformer(seq))
            features = rearrange(features, 'b t c -> t b c')
            visible_count = vis_T
            return features, backward_indexes, visible_count
        else:
            # default: add pos, shuffle, drop masked tokens
            patches = patches + self.pos_embedding
            vis_patches, forward_indexes, backward_indexes, vis_T = self.shuffle(patches)
            seq = torch.cat([self.cls_token.expand(-1, vis_patches.shape[1], -1), vis_patches], dim=0)
            seq = rearrange(seq, 't b c -> b t c')
            features = self.layer_norm(self.transformer(seq))
            features = rearrange(features, 'b t c -> t b c')
            visible_count = vis_T
            return features, backward_indexes, visible_count

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes, visible_count: int = None):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        # If encoder did NOT include mask tokens, add them here. Otherwise skip.
        missing = backward_indexes.shape[0] - features.shape[0]
        if missing > 0:
            features = torch.cat([features, self.mask_token.expand(missing, features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        if visible_count is None:
            mask[T-1:] = 1
        else:
            mask[visible_count:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 mask_sampling: str = 'random',
                 encoder_with_mask_token: bool = False,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, mask_sampling, encoder_with_mask_token)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes, visible_count = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes, visible_count)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
