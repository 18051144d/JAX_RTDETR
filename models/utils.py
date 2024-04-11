import jax
from einops import rearrange
import jax.numpy as jnp
from flax import linen as nn

class identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x
    
def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = jnp.array_split(value, split_shape, axis = 1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = rearrange(value_list[level], 'b (h w) nh c -> (b nh) c h w', h = h, w = w)

        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = rearrange(sampling_grids[:, :, :, level], 'b L n p c -> (b n) L p c')
        x = jax.scipy.ndimage.map_coordinates
        import torch
        torch.nn.functional.grid_sample
        torch.gather
        pass