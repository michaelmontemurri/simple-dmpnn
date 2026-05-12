import torch
import random

# convert undirected edges/features to directed
def make_directed(edges_undir, edge_feat_undir):
    src, rcv, feats = [], [], []
    for i, (u, v) in enumerate(edges_undir):
        src.extend([u, v])
        rcv.extend([v, u])
        feats.append(edge_feat_undir[i])
        feats.append(edge_feat_undir[i])

    edge_index = torch.tensor([src, rcv], dtype=torch.long)
    B = torch.stack(feats).float()
    return edge_index, B


# build adjacency from undirected edge list
def make_adjacency(num_nodes, edges_undir):
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for u, v in edges_undir:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A

# batch a list of graphs
def batch_graphs(graphs):
    """Collate a list of graph dicts into one batched graph representation.

    Args:
        graphs: Iterable of graph dictionaries containing ``X``, ``B``,
            ``edge_index``, and ``y``.

    Returns:
        Tuple ``(X, B, edge_index, rev_index, y, batch_vec)`` where:
            - ``X`` has shape ``[num_nodes_total, node_feat_dim]``
            - ``B`` has shape ``[num_edges_total, edge_feat_dim]``
            - ``edge_index`` has shape ``[2, num_edges_total]``
            - ``rev_index`` has shape ``[num_edges_total]``
            - ``y`` has shape ``[num_graphs, output_dim]``
            - ``batch_vec`` has shape ``[num_nodes_total]``
        assigning each node to a graph in the batch.
    """
    X_list = []
    B_list = []
    edge_index_list = []
    y_list = []
    batch_list = []

    node_offset = 0

    for graph_id, g in enumerate(graphs):
        X = g["X"]
        B = g["B"]
        edge_index = g["edge_index"]
        y = g["y"]

        X_list.append(X)
        B_list.append(B)
        edge_index_list.append(edge_index + node_offset)
        y_list.append(y)
        batch_list.append(torch.full((X.shape[0],), graph_id, dtype=torch.long))

        node_offset += X.shape[0]

    X_batch = torch.vstack(X_list)
    B_batch = torch.vstack(B_list)
    edge_index_batch = torch.cat(edge_index_list, dim=1)
    y_batch = torch.vstack(y_list)          # shape [num_graphs, 1]
    batch_vec = torch.cat(batch_list)       # shape [num_nodes_total]

    rev_index = build_rev_index(edge_index_batch)
    return X_batch, B_batch, edge_index_batch, rev_index, y_batch, batch_vec


def graph_signature(graph):
    X = graph["X"]
    edges = graph["edges_undir"]
    E = graph["E_undir"]

    node_part = tuple(tuple(x.tolist()) for x in X)

    edge_items = []
    for i, (u, v) in enumerate(edges):
        a, b = sorted((u, v))
        edge_items.append((a, b, tuple(E[i].tolist())))

    edge_part = tuple(sorted(edge_items))
    return (node_part, edge_part)


def move_batch_to_device(X, B, edge_index, rev_index, y, batch_vec, device):
    """Move a batched graph tuple to the requested device."""
    return (
        X.to(device),
        B.to(device),
        edge_index.to(device),
        rev_index.to(device),
        y.to(device),
        batch_vec.to(device),
    )

def build_rev_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Build the reverse-edge lookup for a bidirected edge list.

    Args:
        edge_index: Long tensor of shape ``[2, E]`` containing directed edges.

    Returns:
        Long tensor of shape ``[E]`` where position ``i`` stores the index of
        the reverse edge corresponding to ``edge_index[:, i]``.
    """
    src, rcv = edge_index  # shape [E]
    # map (u,v) -> index
    edge_dict = {
        (int(u), int(v)): i
        for i, (u, v) in enumerate(zip(src.tolist(), rcv.tolist()))
    }
    # for each edge (u,v), find index of (v,u)
    rev_index = torch.tensor(
        [edge_dict[(int(v), int(u))] for u, v in zip(src.tolist(), rcv.tolist())],
        dtype=torch.long,
        device=edge_index.device,
    )
    return rev_index

def prepare_batch(graphs, device):
    """Batch graph dicts, move them to device, and package them in a dict."""
    X, B, edge_index, rev_index, y, batch_vec = batch_graphs(graphs)
    X, B, edge_index, rev_index, y, batch_vec = move_batch_to_device(
        X, B, edge_index, rev_index, y, batch_vec, device
    )
    return {
        "X": X,
        "B": B,
        "edge_index": edge_index,
        "rev_index": rev_index,
        "y": y,
        "batch_vec": batch_vec,
        "num_graphs": y.shape[0]
    }

def iter_batches(graphs, batch_size, shuffle=True, rng=None):
    """Yield mini-batches of graph dicts from a Python list dataset."""
    indices = list(range(len(graphs)))
    if shuffle:
        if rng is None:
            random.shuffle(indices)
        else:
            rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield [graphs[i] for i in batch_idx]
