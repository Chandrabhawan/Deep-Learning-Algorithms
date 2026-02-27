"""Generic training utilities: train loop, evaluation, fit function, early stopping."""

import copy
import os
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        checkpoint_path: Path to save the best model weights.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 checkpoint_path: Optional[str] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop. Returns True if patience exhausted."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            if self.checkpoint_path:
                os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: nn.Module) -> None:
        """Restore the best model weights."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        # Squeeze for binary classification or regression
        if output.shape != y_batch.shape and output.ndim > y_batch.ndim:
            output = output.squeeze(-1)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on a data loader. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        if output.shape != y_batch.shape and output.ndim > y_batch.ndim:
            output = output.squeeze(-1)
        loss = loss_fn(output, y_batch)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scheduler: Optional[object] = None,
    early_stopping: Optional[EarlyStopping] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Full training loop with optional validation, early stopping, and LR scheduling.

    Returns:
        Dictionary with keys 'train_loss' and optionally 'val_loss',
        each mapping to a list of per-epoch values.
    """
    history: Dict[str, List[float]] = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        history["train_loss"].append(train_loss)

        msg = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}"

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn, device)
            history["val_loss"].append(val_loss)
            msg += f" | Val Loss: {val_loss:.4f}"

            if early_stopping is not None:
                if early_stopping(val_loss, model):
                    if verbose:
                        print(f"{msg} | Early stopping triggered")
                    early_stopping.load_best(model)
                    break

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(msg)

    return history
