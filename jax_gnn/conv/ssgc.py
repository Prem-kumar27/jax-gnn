import jax
import jax.numpy as jnp
from flax import linen as nn
from ..utils import spmm_scatter


class SSGCLayer(nn.Module):
    """
    Simple spectral graph convolution layer
    https://openreview.net/pdf?id=CYO5T-YjWZV
    """

    out_channels: int
    order: int = 1
    alpha: float = 0.05

    @nn.compact
    def __call__(self, x, edge_indices, edge_attr):
        orig_feature = x
        sum_feature = x
        for _ in range(self.order):
            features = (1 - self.alpha) * spmm_scatter(x, edge_indices, edge_attr)
            sum_feature += features

        feature = sum_feature / self.order + self.alpha * orig_feature
        out = nn.Dense(self.out_channels)(feature)
        return out