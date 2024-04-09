import sys
import math
import copy
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from typing import List

from transformer import get_2d_PositionalEncoding
from conv_block import ConvNormLayer, CSPRepLayer

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union, Sequence

# Seeding for random operations
main_rng = random.PRNGKey(42)

class TransformerEncoderLayer(nn.Module):
    d_model: int
    nhead: int
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: callable = jax.nn.relu
    normalize_before: bool = False

    def setup(self):
        self.self_attn = nn.MultiHeadDotProductAttention(num_heads = self.nhead, qkv_features = self.d_model, dropout_rate = self.dropout)
        # self.self_attn = MultiheadAttention(embed_dim=self.d_model, num_heads=self.nhead, drop_out=self.dropout)
        self.linear1 = nn.Dense(self.dim_feedforward)
        self.dropout0 = nn.Dropout(self.dropout)
        self.linear2 = nn.Dense(self.d_model)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
    
    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def __call__(self, src, src_mask=None, pos_embed=None, train = True):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        print(f'encoder layer: {q.shape}, {k.shape}, {src.shape}, {type(src_mask)}')
        src = self.self_attn(inputs_q = q, inputs_k = k, inputs_v = src, mask=src_mask, deterministic = not train)

        src = residual + self.dropout1(src, deterministic = not train)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout0(self.activation(self.linear1(src)), deterministic=not train))
        src = residual + self.dropout2(src, deterministic=not train)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    num_layers: int = 1
    activation: callable = jax.nn.gelu
    norm = None

    def setup(self):
        self.layers = [TransformerEncoderLayer(d_model = self.d_model, nhead = self.nhead, dim_feedforward = self.dim_feedforward,\
                                                  dropout = self.dropout, activation = self.activation) for _ in range(self.num_layers)]
    
    def __call__(self, src, src_mask=None, pos_embed=None, train = True):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, pos_embed=pos_embed, train = train)
        
        if self.norm:
            src = self.norm(src)

        return src

class HybridEncoder(nn.Module):
    in_channels: Sequence[int]
    hidden_dim: int = 256
    nhead: int = 8
    dim_feedforward: int  = 1024
    dropout: float = 0.0
    enc_act: callable = jax.nn.gelu
    use_encoder_idx = [2]
    num_encoder_layers: int = 1
    pe_temperature: int = 10000
    expansion: float = 1.0
    depth_mult: float = 1.0
    act: callable = jax.nn.silu
    eval_spatial_size = [None, None]
    
    def setup(self):
        # channel projection
        self.input_proj = [nn.Sequential([nn.Conv(self.hidden_dim, kernel_size=(1,1), use_bias = False), nn.BatchNorm(use_running_average=True)]) for _ in self.in_channels]
        
        # encoder transformer
        self.encoder = [TransformerEncoder(d_model=self.hidden_dim,\
                                            nhead=self.nhead, \
                                            dim_feedforward=self.dim_feedforward,
                                            dropout=self.dropout, \
                                            activation=self.enc_act, \
                                            num_layers = self.num_encoder_layers,) \
                                            for _ in range(len(self.use_encoder_idx))]

        # top-down fpn
        lateral_convs = []
        fpn_blocks = []

        for _ in range(len(self.in_channels) - 1 , 0, -1):
            lateral_convs.append(ConvNormLayer(self.hidden_dim, (1, 1), 1, act = self.act))
            fpn_blocks.append(CSPRepLayer(self.hidden_dim, round(3 * self.depth_mult), act = self.act, expansion= self.expansion))

        self.lateral_convs = lateral_convs
        self.fpn_blocks = fpn_blocks

        # bottom-up pan
        downsample_convs = []
        pan_blocks = []
        for _ in range(len(self.in_channels) - 1):
            downsample_convs.append(ConvNormLayer(self.hidden_dim, (3, 3), 2, act = self.act))
            pan_blocks.append(CSPRepLayer(self.hidden_dim, round(3 * self.depth_mult), act = self.act, expansion= self.expansion))

        self.downsample_convs = downsample_convs
        self.pan_blocks = pan_blocks

    def __call__(self, feats, train = True):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        print('proj feat: ',[p.shape for p in proj_feats])
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                b, h, w, c = proj_feats[enc_ind].shape
                # flatten [B, H, W, C] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].reshape((b, h * w, c))
                print('src_flatten: ',src_flatten.shape)
                if train or self.eval_spatial_size is None:
                    pos_embed = get_2d_PositionalEncoding(w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                
                memory = self.encoder[i](src_flatten, pos_embed = pos_embed)
                proj_feats[enc_ind] = memory.reshape((b, h, w, self.hidden_dim))
        
        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            B, H, W, C = feat_high.shape
            upsample_feat = jax.image.resize(feat_high, (B, H * 2, W * 2, C), "nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) -1 -idx](jnp.concatenate([upsample_feat, proj_feats[idx - 1]], axis = - 1))
            inner_outs.insert(0, inner_out)
        
        print('inner_outs: ',[i.shape for i in inner_outs])
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            downsample_feat = self.downsample_convs[idx](outs[-1])
            out = self.pan_blocks[idx](jnp.concatenate([downsample_feat, inner_outs[idx + 1]], axis = -1))
            outs.append(out)
        
        return outs

if __name__ == '__main__':
    setup_time = time.time()
    from backbone import build_resnet
    jax.config.update('jax_platform_name', 'gpu')

    main_rng, x_rng = random.split(main_rng)
    main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)

    # get model
    backbone = build_resnet(size = 18)
    encoder = HybridEncoder(in_channels = [128, 256, 512])

    # init
    x = jax.random.normal(x_rng, (1, 224, 224, 3))
    gpu_x = jax.device_put(x, device=jax.devices('gpu')[0])
    
    bck_time = time.time()
    print(f'Time taken for setup: {bck_time - setup_time}')

    params = backbone.init(init_rng, gpu_x)
    gpu_feats = backbone.apply(params, gpu_x)

    enc_time = time.time()
    print(f'Time taken for backbone: {enc_time - bck_time}')

    params_enc = encoder.init(init_rng, gpu_feats)
    gpu_outputs = encoder.apply(params_enc, gpu_feats, train = True, mutable=['batch_stats'])
    
    print(f'Time taken for encoder: {time.time() - enc_time}')
