"""
Adapters from common external graph formats into this repo's internal graph dict
format used by the DMPNN implementation.

Internal graph dict keys:
    - "X": node features, shape [num_nodes, node_feat_dim]
    - "B": directed edge features, shape [num_directed_edges, edge_feat_dim]
    - "edge_index": directed COO edges, shape [2, num_directed_edges]
    - "y": target tensor, shape [1, output_dim] for single-graph regression

The DMPNN trainer/batching code builds `rev_index` later, so adapters only need
to ensure that every directed edge has a corresponding reverse edge.
"""

from __future__ import annotations

from typing import Iterable

import torch


def _as_2d_target(y) -> torch.Tensor:
    y = torch.as_tensor(y, dtype=torch.float32)
    if y.ndim == 0:
        return y.view(1, 1)
    if y.ndim == 1:
        return y.view(1, -1)
    return y


def _empty_edge_attr(num_edges: int) -> torch.Tensor:
    return torch.zeros((num_edges, 0), dtype=torch.float32)


def _ensure_bidirected(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure every edge (u, v) has a reverse edge (v, u).

    If reverse edges are missing, they are appended with the same edge features.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    edge_index = edge_index.to(torch.long)
    num_edges = edge_index.shape[1]

    if edge_attr is None:
        edge_attr = _empty_edge_attr(num_edges)
    else:
        edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.view(-1, 1)
        if edge_attr.shape[0] != num_edges:
            raise ValueError(
                "edge_attr and edge_index disagree on number of edges: "
                f"{edge_attr.shape[0]} vs {num_edges}"
            )

    src, dst = edge_index
    existing = {(int(u), int(v)) for u, v in zip(src.tolist(), dst.tolist())}

    extra_src = []
    extra_dst = []
    extra_attr = []

    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        if (int(v), int(u)) not in existing:
            extra_src.append(int(v))
            extra_dst.append(int(u))
            extra_attr.append(edge_attr[i].clone())

    if extra_src:
        extra_edge_index = torch.tensor([extra_src, extra_dst], dtype=torch.long)
        extra_edge_attr = torch.stack(extra_attr, dim=0)
        edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, extra_edge_attr], dim=0)

    return edge_index, edge_attr


def from_pyg_data(data, require_y: bool = True) -> dict:
    """Convert a PyG-style ``Data`` object into this repo's graph dict format.

    Expected attributes:
        - ``data.x``
        - ``data.edge_index``
        - optional ``data.edge_attr``
        - optional ``data.y``

    Notes:
        - If ``edge_attr`` is missing, zero-width edge features are created.
        - If reverse directed edges are missing, they are added automatically.
        - If ``require_y=True``, a missing target raises an error.
    """
    if not hasattr(data, "x") or data.x is None:
        raise ValueError("PyG data object must have node features in `data.x`.")
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("PyG data object must have edges in `data.edge_index`.")

    X = torch.as_tensor(data.x, dtype=torch.float32)
    edge_attr = getattr(data, "edge_attr", None)
    edge_index, B = _ensure_bidirected(data.edge_index, edge_attr=edge_attr)

    y = getattr(data, "y", None)
    if y is None:
        if require_y:
            raise ValueError("PyG data object is missing `data.y` and `require_y=True`.")
    else:
        y = _as_2d_target(y)

    graph = {
        "X": X,
        "B": B,
        "edge_index": edge_index,
    }
    if y is not None:
        graph["y"] = y
    return graph


def from_pyg_dataset(dataset: Iterable, require_y: bool = True) -> list[dict]:
    """Convert an iterable of PyG-style ``Data`` objects into graph dicts."""
    return [from_pyg_data(data, require_y=require_y) for data in dataset]
