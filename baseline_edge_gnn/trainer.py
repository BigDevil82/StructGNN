"""
Trainer for Edge-based GNN

Handles training, validation, and evaluation of GNN-EP-4 model.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from baseline_edge_gnn.config import train_config
from baseline_edge_gnn.model import GNNEP4, compute_edge_loss, compute_metrics


class EdgeGNNTrainer:
    """
    Trainer for Edge-based GNN model.
    """

    def __init__(
        self,
        model: GNNEP4,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = None,
        learning_rate: float = None,
        weight_decay: float = None,
    ):
        self.device = device or train_config.DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("CUDA not available, using CPU")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        lr = learning_rate or train_config.LEARNING_RATE
        wd = weight_decay or train_config.WEIGHT_DECAY

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_accuracy": [],
        }

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(batch)

            # Compute loss (L1 as in paper)
            loss = compute_edge_loss(pred, batch.y, batch.edge_mask, loss_type="l1")

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        loader = loader or self.val_loader

        total_loss = 0.0
        all_metrics = {"mae": [], "accuracy": [], "precision": [], "recall": []}
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass
            pred = self.model(batch)

            # Compute loss
            loss = compute_edge_loss(pred, batch.y, batch.edge_mask, loss_type="l1")
            total_loss += loss.item()

            # Compute metrics
            metrics = compute_metrics(pred, batch.y, batch.edge_mask)
            for k, v in metrics.items():
                all_metrics[k].append(v)

            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    def train(
        self,
        epochs: int = None,
        patience: int = None,
        save_path: str = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save best model
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        epochs = epochs or train_config.EPOCHS
        patience = patience or train_config.PATIENCE
        save_path = save_path or train_config.MODEL_SAVE_PATH

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if verbose:
            print(f"Training on {self.device}")
            print(f"Train samples: {len(self.train_loader.dataset)}")
            print(f"Val samples: {len(self.val_loader.dataset)}")
            print("-" * 60)

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    },
                    save_path,
                )

                if verbose:
                    print(f"  [*] New best model saved!")
            else:
                self.patience_counter += 1

            # Print progress
            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"Acc: {val_metrics['accuracy']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Early stopping
            if self.patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if verbose:
                print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

        return self.history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader = None) -> Dict:
        """
        Evaluate model on test set.

        Returns:
            Dictionary of evaluation metrics
        """
        loader = loader or self.test_loader
        if loader is None:
            print("No test loader provided")
            return {}

        self.model.eval()
        loss, metrics = self.validate(loader)

        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print("=" * 60)

        return {"loss": loss, **metrics}

    def plot_history(self, save_path: str = None):
        """Plot training history."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(self.history["train_loss"], label="Train")
        axes[0].plot(self.history["val_loss"], label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (L1)")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Metrics plot
        axes[1].plot(self.history["val_mae"], label="MAE")
        axes[1].plot(self.history["val_accuracy"], label="Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric Value")
        axes[1].set_title("Validation Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training history saved to {save_path}")
        else:
            plt.show()

        plt.close()
