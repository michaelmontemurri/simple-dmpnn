from pathlib import Path

import torch
from .graph_utils import prepare_batch, iter_batches


class DMPNNTrainer:
    """Minimal trainer for graph-level D-MPNN tasks.

    The trainer supports regression, binary classification, and multiclass
    classification through task-aware target formatting and prediction
    postprocessing.
    """
    def __init__(self, model=None, optimizer=None, loss_fn=None, device="cpu", model_class=None, task_type="regression"):
        self.task_type = task_type
        self.model = None if model is None else model.to(device)
        self.model_class = model_class or (None if model is None else model.__class__)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer


    def _forward_model(self, batch):
        """Run the current model on a prepared batch dictionary."""
        return self.model(
            batch["X"],
            batch["B"],
            batch["edge_index"],
            batch["rev_index"],
            batch["batch_vec"],
            batch["num_graphs"],
        )


    def _prepare_targets(self, y, y_hat=None):
        """Format targets to match the configured task type and loss."""
        if self.task_type == "regression":
            return y
        if self.task_type == "binary_classification":
            y = y.float()
            if y_hat is not None:
                return y.view_as(y_hat)
            return y
        if self.task_type == "multiclass_classification":
            return y.view(-1).long()
        raise ValueError(f"Unsupported task_type: {self.task_type}")


    def _compute_loss(self, y_hat, y):
        """Compute task-aware loss from model outputs and raw targets."""
        targets = self._prepare_targets(y, y_hat=y_hat)
        return self.loss_fn(y_hat, targets)


    def _postprocess_predictions(self, y_hat):
        """Convert raw model outputs into task-specific predictions."""
        if self.task_type == "regression":
            return y_hat
        if self.task_type == "binary_classification":
            probs = torch.sigmoid(y_hat)
            return (probs >= 0.5).long()
        if self.task_type == "multiclass_classification":
            return torch.argmax(y_hat, dim=1)
        raise ValueError(f"Unsupported task_type: {self.task_type}")


    def save_checkpoint(self, path, epoch, history, train_loss, val_loss=None):
        """Save model, optimizer, history, and task metadata for resuming."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "task_type": self.task_type,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.get_config() if hasattr(self.model, "get_config") else None,
            "model_class": None if self.model is None else self.model.__class__.__name__,
            "optimizer_state_dict": None if self.optimizer is None else self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "history": history,
            "device": str(self.device),
        }
        torch.save(checkpoint, checkpoint_path)


    def load_checkpoint(self, path):
        """Load a full training checkpoint and restore model state."""
        checkpoint = torch.load(path, map_location=self.device)
        if "task_type" in checkpoint:
            self.task_type = checkpoint["task_type"]

        if self.model is None:
            model_config = checkpoint.get("model_config")
            if self.model_class is None:
                raise ValueError("Cannot reconstruct model from checkpoint without model_class.")
            if model_config is None:
                raise ValueError("Checkpoint does not contain model_config.")
            if hasattr(self.model_class, "from_config"):
                self.model = self.model_class.from_config(model_config).to(self.device)
            else:
                self.model = self.model_class(**model_config).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if self.optimizer is not None and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        return checkpoint


    def train_batch(self, batch):
        """Run one optimization step on a prepared batch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        y_hat = self._forward_model(batch)
        loss = self._compute_loss(y_hat, batch["y"])
        loss.backward()
        self.optimizer.step()
        return loss.detach()
    
    def train_epoch(self, graphs, batch_size, shuffle=True):
        """Train for one epoch and return the graph-weighted average loss."""
        tot_loss = torch.tensor(0.0, device=self.device)
        total_graphs = 0
        for graph_batch in iter_batches(graphs, batch_size, shuffle):
            batch = prepare_batch(graph_batch, self.device)
            batch_loss = self.train_batch(batch)
            batch_graphs = batch["y"].shape[0]
            tot_loss = tot_loss + batch_loss * batch_graphs
            total_graphs += batch_graphs
        return (tot_loss / total_graphs).item()


    @torch.no_grad()
    def evaluate(self, graphs, batch_size):
        """Evaluate the current model and return graph-weighted average loss."""
        self.model.eval()

        tot_loss = torch.tensor(0.0, device=self.device)
        total_graphs = 0

        for graph_batch in iter_batches(graphs, batch_size, shuffle=False):
            batch = prepare_batch(graph_batch, self.device)
            y_hat = self._forward_model(batch)
            batch_loss = self._compute_loss(y_hat, batch["y"])
            batch_graphs = batch["num_graphs"]
            tot_loss = tot_loss + batch_loss.detach() * batch_graphs
            total_graphs += batch_graphs
        avg_loss = tot_loss / total_graphs
        return avg_loss.item()

        
    def fit(
        self,
        train_graphs,
        val_graphs=None,
        epochs=50,
        batch_size=32,
        checkpoint_path=None,
        best_checkpoint_path=None,
        verbose=True,
    ):
        """Train for multiple epochs and optionally evaluate on validation data."""

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = None

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_graphs, batch_size, shuffle=True)
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_graphs is not None:
                val_loss = self.evaluate(val_graphs, batch_size)
                history["val_loss"].append(val_loss)

                if best_checkpoint_path is not None and (
                    best_val_loss is None or val_loss < best_val_loss
                ):
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        best_checkpoint_path,
                        epoch=epoch+1,
                        history=history,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )

            if checkpoint_path is not None:
                self.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    history=history,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )

            if verbose:
                if val_loss is None:
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"train_loss={train_loss:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"train_loss={train_loss:.4f} | "
                        f"val_loss={val_loss:.4f}"
                    )

        return history
    

    def predict(self, graphs, batch_size=32):
        """Generate task-specific predictions for a list of graph dicts."""
        self.model.eval()
        preds = []

        for graph_batch in iter_batches(graphs, batch_size, shuffle=False):
            batch = prepare_batch(graph_batch, self.device)
            y_hat = self._forward_model(batch)
            preds.append(self._postprocess_predictions(y_hat).detach())

        return torch.cat(preds, dim=0).cpu()

    def save_model(self, path):
        """Save a lightweight inference artifact with model weights and config."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "task_type": self.task_type,
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.get_config() if hasattr(self.model, "get_config") else None,
            "model_class": None if self.model is None else self.model.__class__.__name__,
        }
        torch.save(payload, model_path)

    def load_model(self, path):
        """Load a lightweight inference artifact created by ``save_model``."""
        payload = torch.load(path, map_location=self.device)
        if "task_type" in payload:
            self.task_type = payload["task_type"]

        if "model_state_dict" not in payload:
            if self.model is None:
                raise ValueError("A model instance is required to load a raw state_dict payload.")
            self.model.load_state_dict(payload)
            return payload

        if self.model is None:
            model_config = payload.get("model_config")
            if self.model_class is None:
                raise ValueError("Cannot reconstruct model without model_class.")
            if model_config is None:
                raise ValueError("Saved model does not contain model_config.")
            if hasattr(self.model_class, "from_config"):
                self.model = self.model_class.from_config(model_config).to(self.device)
            else:
                self.model = self.model_class(**model_config).to(self.device)

        self.model.load_state_dict(payload["model_state_dict"])
        return payload
