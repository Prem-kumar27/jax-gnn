import jax
import jax.numpy as jnp
from flax import linen as nn
from ..utils import spmm_scatter


class SGCLayer(nn.Module):
    """
    Simple Graph Convolution Layer
    https://arxiv.org/abs/1902.07153
    """

    out_channels: int
    order: int = 1

    @nn.compact
    def __call__(self, x, edge_indices, edge_attr):
        for _ in range(self.order):
            x = spmm_scatter(x, edge_indices, edge_attr)

        x = nn.Dense(self.out_channels)(x)
        return x
