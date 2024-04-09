from functools import partial
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union, Sequence

import jax.numpy as jnp
import flax
from flax import linen as nn

## ================================================Common================================================
ModuleDef = Callable[..., Callable]
# InitFn = Callable[[PRNGKey, Shape, DType], Array]
InitFn = Callable[[Any, Iterable[int], Any], Any]

STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    269: [3, 30, 48, 8],
}

class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.relu
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1
    kernel_init: InitFn = nn.initializers.kaiming_normal()
    bias_init: InitFn = nn.initializers.zeros

    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)

    force_conv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = self.conv_cls(
            self.n_filters,
            self.kernel_size,
            self.strides,
            use_bias=(not self.norm_cls or self.force_conv_bias),
            padding=self.padding,
            feature_group_count=self.groups,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.norm_cls:
            scale_init = (nn.initializers.zeros
                          if self.is_last else nn.initializers.ones)
            mutable = self.is_mutable_collection('batch_stats')
            x = self.norm_cls(use_running_average=not mutable, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x


def slice_variables(variables: Mapping[str, Any],
                    start: int = 0,
                    end: Optional[int] = None) -> flax.core.FrozenDict:
    last_ind = max(int(s.split('_')[-1]) for s in variables['params'])
    if end is None:
        end = last_ind + 1
    elif end < 0:
        end += last_ind + 1

    sliced_variables: Dict[str, Any] = {}
    for k, var_dict in variables.items():  # usually params and batch_stats
        sliced_variables[k] = {
            f'layers_{i-start}': var_dict[f'layers_{i}']
            for i in range(start, end)
            if f'layers_{i}' in var_dict
        }

    return flax.core.freeze(sliced_variables)
## ================================================Common================================================

class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x):
        return self.conv_block_cls(64,
                                   kernel_size=(7, 7),
                                   strides=(2, 2),
                                   padding=[(3, 3), (3, 3)])(x)

class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = self.conv_block_cls(out_shape[-1],
                                    kernel_size=(1, 1),
                                    strides=self.strides,
                                    activation=lambda y: y)(x)
        return x

class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        y = self.conv_block_cls(self.n_hidden,
                                padding=[(1, 1), (1, 1)],
                                strides=self.strides)(x)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)],
                                is_last=True)(y)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))

class ResNetBottleneckBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    groups: int = 1  # cardinality
    base_width: int = 64

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        # Downsampling strides in 3x3 conv instead of 1x1 conv, which improves accuracy.
        # This variant is called ResNet V1.5 (matches torchvision).
        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x)
        y = self.conv_block_cls(group_width,
                                strides=self.strides,
                                groups=self.groups,
                                padding=((1, 1), (1, 1)))(y)
        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y)
        print(y.shape)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))

class ResNet(nn.Module):
    stage_sizes: Sequence[int]
    hidden_sizes: Sequence[int] = (64, 128, 256, 512)
    pool_fn: Callable = partial(nn.max_pool,
                                window_shape=(3, 3),
                                strides=(2, 2),
                                padding=((1, 1), (1, 1)))
    stem_cls: Callable = ResNetStem
    block_cls: Callable = ResNetBlock
    
    def build_stage(self, stage_idx):
        layers = []
        hsize, n_blocks = self.hidden_sizes[stage_idx], self.stage_sizes[stage_idx]
        for b in range(n_blocks):
            strides = (1, 1) if stage_idx == 0 or b != 0 else (2, 2)
            layers.append(self.block_cls(n_hidden=hsize, strides=strides))
        return nn.Sequential(layers)

    def setup(self):
        self.input_proj = nn.Sequential([self.stem_cls() , self.pool_fn])
        self.stages = [self.build_stage(i) for i in range(4)]
    
    def __call__(self, x, train = True):
        outputs = []
        x_ = self.input_proj(x)
        for stage in self.stages:
            x_ = stage(x_)
            outputs.append(x_)
        return outputs[1:]

def build_resnet(size = 18):
    assert size in [18, 34, 50, 101, 152, 200], f'Invalid ResNet size {size}'
    conv_block_cls = partial(ConvBlock, conv_cls = nn.Conv, norm_cls = partial(nn.BatchNorm, momentum=0.9))
    if size < 50:
        return ResNet(stage_sizes=STAGE_SIZES[size], \
                      stem_cls = partial(ResNetStem, conv_block_cls = conv_block_cls), \
                      block_cls = partial(ResNetBlock, conv_block_cls = conv_block_cls))
    else:
        return ResNet(stage_sizes=STAGE_SIZES[size], \
                      stem_cls = partial(ResNetStem, conv_block_cls = conv_block_cls), \
                      block_cls = partial(ResNetBottleneckBlock, conv_block_cls = conv_block_cls))
    
if __name__ == '__main__':
    import jax
    from jax import random

    main_rng = random.PRNGKey(42)
    main_rng, x_rng = random.split(main_rng)
    main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)

    resnet18 = build_resnet(18)
    print('after build')

    # init
    x = jax.random.normal(x_rng, (1, 224, 224, 3))
    params = resnet18.init(init_rng, x)
    print('after init')
    outputs = resnet18.apply(params, x)
    print('after apply')

    print(type(outputs))
    for o in outputs:
        print(o.shape)