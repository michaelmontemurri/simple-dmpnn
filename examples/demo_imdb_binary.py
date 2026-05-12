"""
Train the from-scratch D-MPNN on a non-molecular graph classification dataset.

Dataset:
    IMDB-BINARY from PyTorch Geometric's TUDataset collection.

IMDB-BINARY is a social graph classification dataset, not a molecular dataset.
It does not provide node or edge features, so this script uses the repo's PyG
adapter with:
    - degree one-hot node features
    - constant edge features

Run from the repository root:

    python demo_imdb_binary.py
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch_geometric.datasets import TUDataset

from dmpnn.adapters import from_pyg_dataset
from dmpnn.model import DMPNN
from dmpnn.training import DMPNNTrainer


def set_seed(seed: int) -> None:
    """Set Python and PyTorch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> str:
    """Choose CUDA, Apple Silicon MPS, or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_dataset(
    graphs: list[dict[str, torch.Tensor]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """Create train/validation/test splits."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(graphs), generator=generator).tolist()

    graphs = [graphs[i] for i in perm]

    n_total = len(graphs)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train : n_train + n_val]
    test_graphs = graphs[n_train + n_val :]

    return train_graphs, val_graphs, test_graphs


@torch.no_grad()
def classification_accuracy(
    trainer: DMPNNTrainer,
    graphs: list[dict[str, torch.Tensor]],
    batch_size: int,
) -> float:
    """Compute graph classification accuracy."""
    preds = trainer.predict(graphs, batch_size=batch_size).view(-1)
    y_true = torch.cat([g["y"].view(-1) for g in graphs], dim=0).cpu()

    correct = (preds == y_true).sum().item()
    total = y_true.numel()

    return correct / total


def main() -> None:
    seed = 42
    set_seed(seed)

    device = choose_device()
    print(f"Using device: {device}")

    root = Path("data") / "TUDataset"
    dataset = TUDataset(root=str(root), name="IMDB-BINARY")

    print(f"Loaded dataset: {dataset.name}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")

    graphs = from_pyg_dataset(
        dataset,
        require_y=True,
        task_type="multiclass_classification",
        node_feature_strategy="degree",
        degree_cap=20,
        missing_edge_attr_strategy="constant",
    )

    train_graphs, val_graphs, test_graphs = split_dataset(
        graphs,
        train_frac=0.8,
        val_frac=0.1,
        seed=seed,
    )

    node_feat_dim = graphs[0]["X"].shape[1]
    edge_feat_dim = graphs[0]["B"].shape[1]
    output_size = dataset.num_classes

    print(f"Node feature dim: {node_feat_dim}")
    print(f"Edge feature dim: {edge_feat_dim}")
    print(f"Train/val/test sizes: {len(train_graphs)}, {len(val_graphs)}, {len(test_graphs)}")

    model = DMPNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=128,
        num_steps=3,
        head_hidden_size=128,
        output_size=output_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = DMPNNTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device,
        task_type="multiclass_classification",
    )

    trainer.fit(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        epochs=50,
        batch_size=32,
        verbose=True,
    )

    train_acc = classification_accuracy(trainer, train_graphs, batch_size=32)
    val_acc = classification_accuracy(trainer, val_graphs, batch_size=32)
    test_acc = classification_accuracy(trainer, test_graphs, batch_size=32)

    print()
    print("Final accuracy")
    print(f"Train: {train_acc:.3f}")
    print(f"Val:   {val_acc:.3f}")
    print(f"Test:  {test_acc:.3f}")

    output_path = Path("checkpoints") / "imdb_binary_dmpnn.pt"
    trainer.save_model(output_path)
    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()