import jax
import jax.numpy as jnp


def spmm_scatter(x, edge_indices, edge_attr=None):
    """
    Performs sparse matrix multiplication
    """
    out = x.take(edge_indices[1], axis=0) * jnp.expand_dims(edge_attr, axis=-1)
    out = jax.ops.segment_sum(out, edge_indices[0], x.shape[0])
    return out


def scatter_max(x, index, num_segments=None):
    if num_segments is None:
        num_segments = jnp.max(index) + 1
    out = jnp.full((num_segments,) + x.shape[1:], -jnp.inf)
    return jax.ops.index_max(out, index, x)


def scatter_softmax(edge_indices, edge_values, num_segments):

    maxs = scatter_max(edge_values, edge_indices, num_segments)
    edge_values = edge_values - maxs[edge_indices]
    edge_values = jnp.exp(edge_values)
    out = jnp.zeros((num_segments, edge_values.shape[1]))
    out = jax.ops.index_add(out, edge_indices, edge_values)
    softmax_out = edge_values / (out[edge_indices] + 1e-9)
    return softmax_out


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1):
    num_nodes = jnp.max(edge_index) + 1
    row, col = edge_index[0], edge_index[1]
    mask = row != col

    loop_index = jnp.arange(num_nodes)
    loop_index = jnp.expand_dims(loop_index, 0)
    loop_index = jnp.repeat(loop_index, 2, axis=0)
    edge_index = jnp.concatenate((edge_index[:, mask], loop_index), axis=1)

    if edge_weight is not None:
        loop_weight = jnp.full((num_nodes,), fill_value)

        remaining_edge_weight = edge_weight[~mask]
        if remaining_edge_weight.size > 0:
            loop_weight[row[~mask]] = remaining_edge_weight

        edge_weight = jnp.concatenate([edge_weight[mask], loop_weight], axis=0)

    return edge_index, edge_weight


def symmetric_normalization(num_nodes, edge_index, edge_weight=None):
    if edge_weight is None:
        edge_weight = jnp.ones((edge_index.shape[1]))

    row_sum = jax.ops.segment_sum(edge_weight, edge_index[1], num_nodes)
    row_sum_inv_sqrt = jax.lax.pow(row_sum, -0.5)
    row_sum_inv_sqrt = row_sum_inv_sqrt.at[row_sum_inv_sqrt == float("inf")].set(0)
    return (
        row_sum_inv_sqrt[edge_index[1]] * edge_weight * row_sum_inv_sqrt[edge_index[0]]
    )
