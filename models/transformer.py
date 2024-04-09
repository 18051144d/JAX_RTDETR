import math
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn

## Helper 
class MLP(nn.Module):
    num_layers: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    act: callable = jax.nn.relu

    def setup(self):
        feat_size = [self.hidden_dim] * (self.num_layers - 1) + [self.output_dim]
        self.layers = [nn.Dense(s) for s in feat_size]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers -1 else layer(x)
        return x        

def get_2d_PositionalEncoding(w, h, d_model: int = 256, temperature: float = 10000.0):
    grid_w = jnp.arange(int(w), dtype = np.float32)
    grid_h = jnp.arange(int(h), dtype = np.float32)
    grid_w, grid_h = jnp.meshgrid(grid_w, grid_h ,indexing='ij')
    assert d_model % 4 == 0, \
        'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    
    pos_dim = d_model // 4
    omega = np.arange(pos_dim, dtype = np.float32) / pos_dim
    out_w = jnp.ravel(grid_w)[..., None] @ omega[None]
    out_h = jnp.ravel(grid_h)[..., None] @ omega[None]
    pe = jnp.concatenate([jnp.sin(out_w), jnp.cos(out_w), jnp.sin(out_h), jnp.cos(out_h)], axis = 1)[None, :, :]
    pe = jax.device_put(pe)
    return pe



## Transformer Modules
class MSDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module
    """
    embed_dim: int = 256
    num_heads: int = 8
    num_levels: int = 4
    num_points: int = 4

    def setup(self):
        self.total_points = self.num_heads * self.num_levels * self.num_points
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Dense(self.total_points * 2,)
        self.attention_weights = nn.Dense(self.total_points)
        self.value_proj = nn.Dense(self.embed_dim)
        self.output_proj = nn.Dense(self.embed_dim)

        # self.ms_deformable_attn_core = deformable_attention_core_func

        # self._reset_parameters()
    
    def __call__(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q, _ = query.shape
        _, Len_v, _ = value.shape

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = jnp.expand_dims(value_mask.astype(value.dtype), dim = -1)
            value *= value_mask
        
        