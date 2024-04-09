from flax import linen as nn

class identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x