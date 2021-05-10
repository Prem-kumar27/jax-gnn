import jax
import jax.numpy as jnp
from flax import linen as nn
from ..utils import scatter_softmax


class TransformerConv(nn.Module):
    """
    https://arxiv.org/pdf/2009.03509.pdf
    """

    out_channels: int
    n_heads: int = 4
    concat: bool = True
    feat_drop: float = 0.5
    attn_drop: float = 0.5
    residual: bool = True
    gate: bool = True

    @nn.compact
    def __call__(self, x, edge_indices, edge_attr=None, train=True):

        n_nodes = x.shape[0]
        src_index, trg_index = edge_indices

        if self.feat_drop:
            x = nn.Dropout(rate=self.feat_drop)(x, deterministic=not train)

        x_i = jnp.take(x, src_index, axis=0)
        x_j = jnp.take(x, trg_index, axis=0)

        query = nn.Dense(self.out_channels * self.n_heads)(x_i).reshape(
            -1, self.n_heads, self.out_channels
        )
        key = nn.Dense(self.out_channels * self.n_heads)(x_j).reshape(
            -1, self.n_heads, self.out_channels
        )

        if edge_attr is not None:
            edge_proj = nn.Dense(self.out_channels * self.n_heads, use_bias=False)(
                edge_attr
            ).reshape(-1, self.n_heads, self.out_channels)
            key += edge_proj

        alpha = (query * key).sum(axis=-1) / jnp.sqrt(self.out_channels)
        alpha = scatter_softmax(trg_index, alpha, n_nodes)

        if self.attn_drop:
            alpha = nn.Dropout(rate=self.attn_drop)(alpha, deterministic=not train)

        val = nn.Dense(self.out_channels * self.n_heads)(x_j).reshape(
            -1, self.n_heads, self.out_channels
        )

        if edge_attr is not None:
            val += edge_proj

        out = val * jnp.expand_dims(alpha, axis=-1)

        # Aggregate
        node_feat = jnp.zeros((n_nodes, *out.shape[1:]))
        node_feat = jax.ops.index_add(node_feat, trg_index, out)

        if self.concat:
            node_feat = node_feat.reshape(-1, self.n_heads * self.out_channels)
        else:
            node_feat = node_feat.mean(axis=1)

        if self.residual:
            if self.concat:
                skip_feat = nn.Dense(self.n_heads * self.out_channels)(x)
            else:
                skip_feat = nn.Dense(self.out_channels)(x)

            if self.gate:
                gate = nn.Dense(1)(
                    jnp.concatenate(
                        [node_feat, skip_feat, skip_feat - node_feat], axis=-1
                    )
                )
                gate = jax.nn.sigmoid(gate)
                node_feat = gate * skip_feat + (1 - gate) * node_feat
            else:
                node_feat = node_feat + skip_feat

        return node_feat
