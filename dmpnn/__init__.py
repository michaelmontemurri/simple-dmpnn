"""Public package interface for the reusable D-MPNN implementation."""

from .adapters import from_pyg_data, from_pyg_dataset
from .model import DMPNN, DMPNNEncoder
from .training import DMPNNTrainer

__all__ = [
    "DMPNN",
    "DMPNNEncoder",
    "DMPNNTrainer",
    "from_pyg_data",
    "from_pyg_dataset",
]

__version__ = "0.1.0"
