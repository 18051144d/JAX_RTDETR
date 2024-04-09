import math
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn


# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

def scaled_dot_product(q, k, v, dropout = 0.1, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    attention = nn.Dropout(dropout)(attention)
    values = jnp.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    embed_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads (h)
    drop_out: float

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None, train=True):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, drop_out = self.drop_out, mask=mask)
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention

class PositionalEncoding(nn.Module):
    d_model : int         # Hidden dimensionality of the input.
    max_len : int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:,None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x, train=True):
        x = x + self.pe[:, :x.shape[1]]
        return x

class PositionalEncoding2D(nn.Module):
    w : int
    h : int
    d_model : int= 256
    temperature: float = 10000.0

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