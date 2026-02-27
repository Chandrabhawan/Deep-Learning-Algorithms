"""Classification and regression metrics helpers."""

from typing import Dict, Optional

import numpy as np
import torch


@torch.no_grad()
def compute_accuracy(model, loader, device) -> float:
    """Compute classification accuracy over a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        if output.ndim == 1 or output.shape[-1] == 1:
            preds = (torch.sigmoid(output.squeeze(-1)) > 0.5).long()
        else:
            preds = output.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return correct / max(total, 1)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
) -> Dict:
    """Thin wrapper around sklearn classification_report returning a dict."""
    from sklearn.metrics import classification_report
    return classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
