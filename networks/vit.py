import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.unetr import UNETR
from functools import partial
from timm.models.vision_transformer import Block
from timm.models.layers import DropPath, to_2tuple

torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.')

def build_2d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    """
    TODO: the code can work when grid size is isotropic (H==W), but it is not logically right especially when data is non-isotropic(H!=W).
    """
    h, w = grid_size, grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if is_torch2:
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if is_torch2:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop,
            )
            x = attn.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        if is_torch2:
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
        query from decoder (x), key and value from encoder (y)
        """
        B, N, C = x.shape
        Ny = y.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if is_torch2:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop,
            )
            x = attn.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, self_attn=False):
        super().__init__()
        print("CrossAttentionBlock")
        self.self_attn = self_attn
        if self.self_attn:
            self.norm0 = norm_layer(decoder_dim)
            self.self_attn = Attention(
                decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(decoder_dim)
        self.cross_attn = CrossAttention(
            encoder_dim, decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(decoder_dim)
        mlp_hidden_dim = int(decoder_dim * mlp_ratio)
        self.mlp = Mlp(in_features=decoder_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        """
        x: decoder feature; y: encoder feature (after layernorm)
        """
        
        if self.self_attn:
            x = x + self.drop_path(self.self_attn(self.norm0(x)))
        x = x + self.drop_path(self.cross_attn(self.norm1(x), y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MAEViTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """
    def __init__(self, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=None, norm_layer=None, act_layer=None, use_pe=True, return_patchembed=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 # don't consider distillation here
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.use_pe = use_pe
        self.return_patchembed = return_patchembed

        self.patch_embed = embed_layer(img_size=patch_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        assert self.patch_embed.num_patches == 1, \
                "Current embed layer should output 1 token because the patch length is reshaped to batch dimension"

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_pe = nn.Parameter(torch.zeros([1, 1, embed_dim], dtype=torch.float32))
        # self.cls_pe.requires_grad = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # init patch embed parameters
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_token, std=.02, a=-.02, b=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward_features(self, x, pos_embed=None):
        return_patchembed = self.return_patchembed

        embed_dim = self.embed_dim
        B, L, _ = x.shape

        x = self.patch_embed(x) # [B*L, embed_dim]
        x = x.reshape(B, L, embed_dim)
        if return_patchembed:
            patchembed = x
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        if self.use_pe:
            if x.size(1) != pos_embed.size(1):
                assert x.size(1) == pos_embed.size(1) + 1, "Unmatched x and pe shapes"
                cls_pe = torch.zeros([B, 1, embed_dim], dtype=torch.float32).to(x.device)
                pos_embed = torch.cat([cls_pe, pos_embed], dim=1)
            x = self.pos_drop(x + pos_embed)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        x = self.norm(x)
        if return_patchembed:
            return x, patchembed, hidden_states_out
        else:
            return x, hidden_states_out

    def forward(self, x, pos_embed=None):
        if self.return_patchembed:
            x, patch_embed, hidden_states_out = self.forward_features(x, pos_embed)
        else:
            x, hidden_states_out = self.forward_features(x, pos_embed)
        x = self.head(x)
        if self.return_patchembed:
            return x, patch_embed, hidden_states_out
        else:
            return x, hidden_states_out

class MAEViTDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, self_depth=3, cross_depth=3,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None, self_attn=False):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * patch_size ** 2
        self.embed_dim = embed_dim
        self.num_tokens = 1 # don't consider distillation here
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self_depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self_depth)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, cross_depth)]  # stochastic depth decay rule
        self.cross_attention_blocks = nn.ModuleList([CrossAttentionBlock(encoder_dim=embed_dim, decoder_dim=embed_dim,
                                                          num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None,
                                                          drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                                                          norm_layer=norm_layer, act_layer=act_layer, self_attn=self_attn) for i in range(cross_depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, sel_dic):

        x = sel_dic[12]
        e9 = sel_dic[9]
        e6 = sel_dic[6]
        e3 = sel_dic[3]
        
        for blk in self.blocks:
            x = blk(x)
        x = self.cross_attention_blocks[0](x, e9)
        x = self.cross_attention_blocks[1](x, e6)
        x = self.cross_attention_blocks[2](x, e3)
        x = self.norm(x)
        return x

    def forward(self, sel_lists):

        x = self.forward_features(sel_lists)
        x = self.head(x)
        return x

def vit_base_patch16_96(**kwargs):
    model = MAEViTEncoder(
        embed_dim=768,
        num_heads=12,
        depth=12,
        **kwargs)
    return model

def decoder_base_patch16_96(**kwargs):
    model = MAEViTDecoder(
        embed_dim=384,
        self_depth=2,
        cross_depth=3,
        num_heads=12,
        **kwargs)
    return model

def vit_large_patch16_96(**kwargs):
    model = MAEViTEncoder(
        embed_dim=1536,
        num_heads=16,
        depth=12,
        **kwargs)
    return model

def decoder_large_patch16_96(**kwargs):
    model = MAEViTDecoder(
        embed_dim=528,
        self_depth=2,
        cross_depth=3,
        num_heads=16,
        **kwargs)
    return model

def vit_large_patch12_96(**kwargs):
    model = MAEViTEncoder(
        embed_dim=1536,
        num_heads=16,
        depth=12,
        **kwargs)
    return model

def decoder_large_patch12_96(**kwargs):
    model = MAEViTDecoder(
        embed_dim=528,
        self_depth=2,
        cross_depth=3,
        num_heads=16,
        **kwargs)
    return model
