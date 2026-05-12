# A script for training a DMPNN on a synthetically generated set of graphs with a
# known response value based on the edge and bond features.
import torch
from dataclasses import dataclass

from examples.synthetic_graph_gen import generate_unique_graphs
from dmpnn.model import DMPNN
from dmpnn.training import DMPNNTrainer

@dataclass
class TrainingConfig:
    node_feat_dim: int = 4
    edge_feat_dim: int = 2
    hidden_dim: int = 128
    num_steps: int = 3
    head_hidden_size: int = 128
    output_size: int = 1
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 100


def main():
    config = TrainingConfig()
    train_graphs, train_sigs = generate_unique_graphs(10000, min_nodes=3, max_nodes=10, seed=42)
    val_graphs, _ = generate_unique_graphs(2000, min_nodes=3, max_nodes=10, existing_sigs=train_sigs, seed=43)

    model = DMPNN(
            config.node_feat_dim,
            config.edge_feat_dim,
            config.hidden_dim,
            config.num_steps,
            config.head_hidden_size,
            config.output_size
    )
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    loss_fn = torch.nn.MSELoss()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using {device}")

    trainer = DMPNNTrainer(
        model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )


    history = trainer.fit(train_graphs, val_graphs, config.epochs, config.batch_size)
    trainer.save_model("saved_models/dmpnn_regressor.pt")
    print(history)

if __name__ == "__main__":
    main()
