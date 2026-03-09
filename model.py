import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MAEViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        enc_dim=768,
        enc_depth=12,
        enc_heads=12,
        dec_dim=384,
        dec_depth=12,
        dec_heads=6,
        mask_ratio=0.75,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_dim)
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = patch_size * patch_size * 3

        self.enc_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, enc_dim))
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(enc_dim, enc_heads) for _ in range(enc_depth)]
        )
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dec_dim))
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(dec_dim, dec_heads) for _ in range(dec_depth)]
        )
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, self.patch_dim)

    def unpatchify(self, patches):
        p = self.patch_size
        b, n, _ = patches.shape
        h = w = int(math.sqrt(n))
        x = patches.reshape(b, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, 3, h * p, w * p)

    def forward(self, imgs):

        x = self.patch_embed(imgs) + self.enc_pos_embed

        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.enc_norm(x)
        x = self.enc_to_dec(x)

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.dec_norm(x)
        pred = self.dec_pred(x)

        # generate MAE mask
        B, N, _ = pred.shape
        num_mask = int(self.mask_ratio * N)

        mask = torch.zeros(B, N, device=pred.device)

        for b in range(B):
            ids = torch.randperm(N, device=pred.device)[:num_mask]
            mask[b, ids] = 1

        return None, pred, mask, None