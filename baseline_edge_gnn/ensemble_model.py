"""
Ensemble model for K-fold cross-validation results.

Loads multiple models from K-fold CV and averages their predictions.
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from baseline_edge_gnn.model import GNNEP4


class EnsembleGNNEP4(nn.Module):
    """
    Ensemble wrapper for multiple GNNEP4 models.

    Loads models from K-fold CV and averages predictions.
    """

    def __init__(
        self,
        model_paths: List[str],
        node_in_dim: int,
        edge_in_dim: int,
        gnn_out_dim: int = 128,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize ensemble model.

        Args:
            model_paths: List of paths to saved model checkpoints
            node_in_dim: Node feature dimension
            edge_in_dim: Edge feature dimension
            gnn_out_dim: GNN output dimension
            dropout: Dropout rate
            device: Device to load models on
        """
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = nn.ModuleList()

        # Load all models
        for path in model_paths:
            if not os.path.exists(path):
                print(f"Warning: Model not found at {path}, skipping...")
                continue

            model = GNNEP4(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                gnn_out_dim=gnn_out_dim,
                dropout=dropout,
            )

            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            self.models.append(model)

        print(f"Loaded {len(self.models)} models for ensemble prediction")

        if len(self.models) == 0:
            raise ValueError("No valid models loaded for ensemble!")

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass with ensemble averaging.

        Args:
            data: Batch of graph data

        Returns:
            Averaged edge predictions (shape: [num_edges, 1])
        """
        data = data.to(self.device)

        # Collect predictions from all models
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(data)
                predictions.append(pred)

        # Average predictions
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)

        return ensemble_pred

    @classmethod
    def from_cv_dir(
        cls,
        cv_dir: str,
        node_in_dim: int,
        edge_in_dim: int,
        gnn_out_dim: int = 128,
        dropout: float = 0.1,
        n_folds: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Load ensemble from cross-validation directory.

        Args:
            cv_dir: Directory containing fold_1.pt, fold_2.pt, etc.
            node_in_dim: Node feature dimension
            edge_in_dim: Edge feature dimension
            gnn_out_dim: GNN output dimension
            dropout: Dropout rate
            n_folds: Number of folds (if None, auto-detect)
            device: Device to load models on

        Returns:
            EnsembleGNNEP4 instance
        """
        model_paths = []

        if n_folds is None:
            # Auto-detect number of folds
            fold_num = 1
            while os.path.exists(os.path.join(cv_dir, f"fold_{fold_num}.pt")):
                model_paths.append(os.path.join(cv_dir, f"fold_{fold_num}.pt"))
                fold_num += 1
        else:
            # Use specified number of folds
            for i in range(1, n_folds + 1):
                path = os.path.join(cv_dir, f"fold_{i}.pt")
                if os.path.exists(path):
                    model_paths.append(path)

        if len(model_paths) == 0:
            raise ValueError(f"No fold models found in {cv_dir}")

        print(f"Found {len(model_paths)} fold models in {cv_dir}")

        return cls(
            model_paths=model_paths,
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            gnn_out_dim=gnn_out_dim,
            dropout=dropout,
            device=device,
        )
