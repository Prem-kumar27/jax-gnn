import jax
import jax.numpy as jnp
from flax import linen as nn
from ..utils import scatter_softmax


class GATLayer(nn.Module):
    """
    Graph Attention Layer
    https://arxiv.org/abs/1710.10903
    """

    out_channels: int
    n_heads: int
    dropout_rate: float
    concat: bool = False
    residual: bool = False
    use_bias: bool = False
    source_nodes_dim = 0
    target_nodes_dim = 1

    @nn.compact
    def __call__(self, x, edge_indices, edge_att=None, train=True):

        n_nodes = x.shape[0]
        a_target = self.param(
            "a_target", nn.initializers.zeros, (1, self.n_heads, self.out_channels)
        )
        a_source = self.param(
            "a_source", nn.initializers.zeros, (1, self.n_heads, self.out_channels)
        )

        if self.use_bias and self.concat:
            bias = self.param(
                "bias", nn.initializers.zeros, (self.n_heads * self.out_channels,)
            )
        elif self.use_bias and not self.concat:
            bias = self.param("bias", nn.initializers.zeros, (self.out_channels,))

        linear_trans = nn.Dense(self.n_heads * self.out_channels, use_bias=False)(x)
        linear_trans = linear_trans.reshape((-1, self.n_heads, self.out_channels))
        linear_trans = nn.Dropout(rate=self.dropout_rate)(
            linear_trans, deterministic=not train
        )

        hidden_target = (a_target * linear_trans).sum(axis=-1)[
            edge_indices[self.target_nodes_dim, :]
        ]
        hidden_source = (a_source * linear_trans).sum(axis=-1)[
            edge_indices[self.source_nodes_dim, :]
        ]
        node_features_selected = linear_trans[edge_indices[self.source_nodes_dim, :]]

        edge_scores = nn.leaky_relu(hidden_source + hidden_target)
        edge_attention = scatter_softmax(
            edge_indices[self.target_nodes_dim], edge_scores, n_nodes
        )
        edge_attention = nn.Dropout(rate=self.dropout_rate)(
            edge_attention, deterministic=not train
        )

        node_features_selected_weighted = node_features_selected * jnp.expand_dims(
            edge_attention, axis=-1
        )
        node_feat = jnp.zeros((n_nodes, *node_features_selected_weighted.shape[1:]))
        node_feat = jax.ops.index_add(
            node_feat,
            edge_indices[self.target_nodes_dim],
            node_features_selected_weighted,
        )

        if self.residual:
            node_feat += nn.Dense(self.n_heads * self.out_channels, use_bias=False)(
                x
            ).reshape((-1, self.n_heads, self.out_channels))

        if self.concat:
            node_feat = node_feat.reshape(-1, self.n_heads * self.out_channels)
        else:
            node_feat = node_feat.mean(axis=1)

        if self.use_bias:
            node_feat = node_feat + bias

        return node_feat