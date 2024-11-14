import numpy as np
import torch
import torch.nn as nn
from timm.models.layers.helpers import to_3tuple
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


# def get_3d_sincos_position_embedding(embed_dim, grid_size, num_tokens=1, temperature=10000.):
#
#     grid_size = to_3tuple(grid_size)
#     h, w, d = grid_size
#     grid_h = torch.arange(h, dtype=torch.float32)
#     grid_w = torch.arange(w, dtype=torch.float32)
#     grid_d = torch.arange(d, dtype=torch.float32)
#
#     grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
#     assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
#     pos_dim = embed_dim // 6
#     omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
#     omega = 1. / (temperature**omega)
#     out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
#     out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
#     out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
#     pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]
#
#     assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
#     if num_tokens == 1:
#         pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
#         pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
#     else:
#         pos_embed = nn.Parameter(pos_emb)
#     pos_embed.requires_grad = False
#     return pos_embed


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_d = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*D, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*D, D/3)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*D, D/3)
    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1) # (H*W*D, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


