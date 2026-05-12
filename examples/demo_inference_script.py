import torch
from examples.synthetic_graph_gen import generate_unique_graphs
from dmpnn.model import DMPNN
from dmpnn.training import DMPNNTrainer


def main():
    test_graphs, _ = generate_unique_graphs(100, min_nodes=3, max_nodes=10, seed=42)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using {device}")
    
    trainer = DMPNNTrainer(
        model=None,
        model_class=DMPNN,
        loss_fn=torch.nn.MSELoss(),
        device=device,
    )
    trainer.load_model("saved_models/dmpnn_regressor.pt")

    y_hat = trainer.predict(test_graphs, batch_size=32)
    y_true = torch.vstack([graph["y"] for graph in test_graphs])

    mse = trainer.evaluate(test_graphs, batch_size=32)
    mae = torch.mean(torch.abs(y_hat - y_true)).item()

    print("First 10 predictions:")
    for idx, (y_t, y_p) in enumerate(zip(y_true[:10], y_hat[:10])):
        print(f"  graph {idx:>2}: true = {y_t.item():>6.2f} | pred = {y_p.item():>6.2f}")

    print(f"\nInference set size: {len(test_graphs)}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    main()

