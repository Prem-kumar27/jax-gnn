import jax
import jax.numpy as jnp
from flax import linen as nn

from ..conv.gcn import GCNLayer
from ..utils import symmetric_normalization, add_remaining_self_loops


class GCNModel(nn.Module):
    n_out: int

    @nn.compact
    def __call__(self, inputs):
        x, edge_indices, edge_attr = inputs

        edge_indices, edge_attr = add_remaining_self_loops(edge_indices)
        edge_attr = symmetric_normalization(x.shape[0], edge_indices, edge_attr)

        out = GCNLayer(out_channels=16)(x, edge_indices, edge_attr)
        x = nn.relu(x)
        out = GCNLayer(out_channels=self.n_out)(out, edge_indices, edge_attr)
        out = nn.log_softmax(out)
        return out
