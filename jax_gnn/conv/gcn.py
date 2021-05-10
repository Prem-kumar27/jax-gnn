import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from ..utils import spmm_scatter


class GCNLayer(nn.Module):
    """
    Graph Convolution layer
    https://arxiv.org/abs/1609.02907
    """

    out_channels: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.kaiming_uniform()
    bias_init: Callable = nn.initializers.zeros
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, edge_indices, edge_attr):
        input_dim = x.shape[-1]
        kernel = self.param("kernel", self.kernel_init, (input_dim, self.out_channels))
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.out_channels,))

        out = jnp.dot(x, kernel)
        out = spmm_scatter(out, edge_indices, edge_attr)
        if self.use_bias:
            out = out + bias

        return out