"""
Adapters from common external graph formats into this repo's internal graph dict
format used by the DMPNN implementation.

Internal graph dict keys:
    - "X": node features, shape [num_nodes, node_feat_dim]
    - "B": directed edge features, shape [num_directed_edges, edge_feat_dim]
    - "edge_index": directed COO edges, shape [2, num_directed_edges]
    - "y": optional target tensor

For graph-level regression:
    - y is formatted as float tensor, shape [1, output_dim]

For graph-level classification:
    - y is formatted as long tensor containing class indices

The DMPNN trainer/batching code builds `rev_index` later, so adapters only need
to ensure that every directed edge has a corresponding reverse edge.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


SUPPORTED_TASK_TYPES = {
    "regression",
    "binary_classification",
    "multiclass_classification",
}


def _format_target(y, task_type: str = "regression") -> torch.Tensor:
    """Format a graph-level target for the configured task type."""
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type: {task_type}. "
            f"Expected one of {sorted(SUPPORTED_TASK_TYPES)}."
        )

    y = torch.as_tensor(y)

    if task_type == "regression":
        y = y.float()
        if y.ndim == 0:
            return y.view(1, 1)
        if y.ndim == 1:
            return y.view(1, -1)
        return y

    if task_type == "binary_classification":
        # For BCEWithLogitsLoss, users may prefer float targets.
        # For CrossEntropyLoss with two classes, use multiclass_classification.
        return y.view(-1).float()

    if task_type == "multiclass_classification":
        return y.view(-1).long()

    raise ValueError(f"Unsupported task_type: {task_type}")


def _empty_edge_attr(num_edges: int) -> torch.Tensor:
    """Create zero-width edge features."""
    return torch.zeros((num_edges, 0), dtype=torch.float32)


def _constant_edge_attr(num_edges: int, value: float = 1.0) -> torch.Tensor:
    """Create a single constant edge feature."""
    return torch.full((num_edges, 1), fill_value=value, dtype=torch.float32)


def _degree_one_hot_from_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    degree_cap: int = 20,
) -> torch.Tensor:
    """Build one-hot node features from node degree.

    This is useful for unattributed graph datasets such as IMDB-BINARY.
    Degrees greater than degree_cap are clipped into the final bucket.
    """
    edge_index = edge_index.to(torch.long)

    if edge_index.numel() == 0:
        degree = torch.zeros(num_nodes, dtype=torch.long)
    else:
        src = edge_index[0]
        degree = torch.zeros(num_nodes, dtype=torch.long)
        degree.scatter_add_(0, src.cpu(), torch.ones_like(src.cpu()))

    degree = degree.clamp(max=degree_cap)
    return F.one_hot(degree, num_classes=degree_cap + 1).float()


def _ensure_node_features(
    data,
    node_feature_strategy: str = "require",
    degree_cap: int = 20,
) -> torch.Tensor:
    """Return node features from a PyG Data object.

    Args:
        data: PyG-style Data object.
        node_feature_strategy:
            - "require": require data.x to exist.
            - "degree": use data.x if present, otherwise create degree one-hot features.
            - "constant": use data.x if present, otherwise create one constant feature per node.
        degree_cap: Maximum degree bucket for degree one-hot features.
    """
    if hasattr(data, "x") and data.x is not None:
        return torch.as_tensor(data.x, dtype=torch.float32)

    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        raise ValueError(
            "PyG data object does not have `data.x` or a usable `data.num_nodes`."
        )

    num_nodes = int(data.num_nodes)

    if node_feature_strategy == "require":
        raise ValueError(
            "PyG data object must have node features in `data.x`. "
            "For unattributed datasets, use node_feature_strategy='degree' "
            "or node_feature_strategy='constant'."
        )

    if node_feature_strategy == "degree":
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("Degree features require `data.edge_index`.")
        return _degree_one_hot_from_edge_index(
            data.edge_index,
            num_nodes=num_nodes,
            degree_cap=degree_cap,
        )

    if node_feature_strategy == "constant":
        return torch.ones((num_nodes, 1), dtype=torch.float32)

    raise ValueError(
        f"Unsupported node_feature_strategy: {node_feature_strategy}. "
        "Expected one of {'require', 'degree', 'constant'}."
    )


def _ensure_edge_attr(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    missing_edge_attr_strategy: str = "zeros",
) -> torch.Tensor:
    """Return edge features aligned with edge_index.

    Args:
        edge_index: Tensor of shape [2, num_edges].
        edge_attr: Optional edge attributes.
        missing_edge_attr_strategy:
            - "zeros": create zero-width edge features, shape [E, 0].
            - "constant": create one constant edge feature, shape [E, 1].
    """
    num_edges = edge_index.shape[1]

    if edge_attr is None:
        if missing_edge_attr_strategy == "zeros":
            return _empty_edge_attr(num_edges)
        if missing_edge_attr_strategy == "constant":
            return _constant_edge_attr(num_edges)
        raise ValueError(
            f"Unsupported missing_edge_attr_strategy: {missing_edge_attr_strategy}. "
            "Expected one of {'zeros', 'constant'}."
        )

    edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.view(-1, 1)

    if edge_attr.shape[0] != num_edges:
        raise ValueError(
            "edge_attr and edge_index disagree on number of edges: "
            f"{edge_attr.shape[0]} vs {num_edges}"
        )

    return edge_attr


def _ensure_bidirected(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    missing_edge_attr_strategy: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure every directed edge (u, v) has a reverse edge (v, u).

    If reverse edges are missing, they are appended with the same edge features.

    Duplicate input edges are preserved. This function guarantees that for every
    input edge, at least one reverse edge exists. It does not deduplicate the
    full edge list.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
        )

    edge_index = edge_index.to(torch.long).cpu()
    edge_attr = _ensure_edge_attr(
        edge_index,
        edge_attr=edge_attr,
        missing_edge_attr_strategy=missing_edge_attr_strategy,
    ).cpu()

    src, dst = edge_index
    existing = {(int(u), int(v)) for u, v in zip(src.tolist(), dst.tolist())}

    extra_edges: list[tuple[int, int]] = []
    extra_attr: list[torch.Tensor] = []

    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        u = int(u)
        v = int(v)

        if u == v:
            # Self-loop is its own reverse.
            continue

        if (v, u) not in existing:
            extra_edges.append((v, u))
            extra_attr.append(edge_attr[i].clone())
            existing.add((v, u))

    if extra_edges:
        extra_edge_index = torch.tensor(extra_edges, dtype=torch.long).t().contiguous()
        extra_edge_attr = torch.stack(extra_attr, dim=0)

        edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, extra_edge_attr], dim=0)

    return edge_index, edge_attr


def from_pyg_data(
    data,
    require_y: bool = True,
    task_type: str = "regression",
    node_feature_strategy: str = "require",
    degree_cap: int = 20,
    missing_edge_attr_strategy: str = "zeros",
) -> dict:
    """Convert a PyG-style Data object into this repo's graph dict format.

    Expected PyG attributes:
        - data.edge_index
        - optional data.x
        - optional data.edge_attr
        - optional data.y

    Args:
        data: PyG-style Data object.
        require_y: If True, raise an error when data.y is missing.
        task_type: One of {"regression", "binary_classification",
            "multiclass_classification"}.
        node_feature_strategy:
            - "require": require data.x.
            - "degree": use data.x if present; otherwise create one-hot degree features.
            - "constant": use data.x if present; otherwise create constant node features.
        degree_cap: Maximum degree bucket for degree one-hot features.
        missing_edge_attr_strategy:
            - "zeros": use zero-width edge features if edge_attr is missing.
            - "constant": use one constant edge feature if edge_attr is missing.

    Returns:
        graph: Dictionary with keys "X", "B", "edge_index", and optionally "y".
    """
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type: {task_type}. "
            f"Expected one of {sorted(SUPPORTED_TASK_TYPES)}."
        )

    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("PyG data object must have edges in `data.edge_index`.")

    X = _ensure_node_features(
        data,
        node_feature_strategy=node_feature_strategy,
        degree_cap=degree_cap,
    )

    edge_attr = getattr(data, "edge_attr", None)
    edge_index, B = _ensure_bidirected(
        data.edge_index,
        edge_attr=edge_attr,
        missing_edge_attr_strategy=missing_edge_attr_strategy,
    )

    y = getattr(data, "y", None)
    if y is None:
        if require_y:
            raise ValueError("PyG data object is missing `data.y` and `require_y=True`.")
    else:
        y = _format_target(y, task_type=task_type)

    graph = {
        "X": X,
        "B": B,
        "edge_index": edge_index,
    }

    if y is not None:
        graph["y"] = y

    return graph


def from_pyg_dataset(
    dataset: Iterable,
    require_y: bool = True,
    task_type: str = "regression",
    node_feature_strategy: str = "require",
    degree_cap: int = 20,
    missing_edge_attr_strategy: str = "zeros",
) -> list[dict]:
    """Convert an iterable of PyG-style Data objects into graph dicts."""
    return [
        from_pyg_data(
            data,
            require_y=require_y,
            task_type=task_type,
            node_feature_strategy=node_feature_strategy,
            degree_cap=degree_cap,
            missing_edge_attr_strategy=missing_edge_attr_strategy,
        )
        for data in dataset
    ]