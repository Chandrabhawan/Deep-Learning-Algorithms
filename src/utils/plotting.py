"""Plotting utilities for training curves, confusion matrices, and ROC/PR curves."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    history: Dict[str, List[float]],
    title: str = "Training & Validation Loss",
    figsize: tuple = (8, 4),
) -> None:
    """Plot training (and optionally validation) loss curves.

    Args:
        history: Dict with 'train_loss' and optionally 'val_loss'.
    """
    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (6, 5),
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix using matplotlib (no seaborn dependency)."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    figsize: tuple = (12, 4),
) -> None:
    """Plot ROC and Precision-Recall curves side by side (binary classification)."""
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, average_precision_score
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    ax2.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
