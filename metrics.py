"""
Metrics utilities for DeepFake Detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from typing import List, Optional


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> dict:
    """
    Compute standard binary classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, cm
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "cm":        confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    return metrics


def plot_confusion_matrix(
    cm,
    class_names: List[str] = ["Real", "Fake"],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
):
    """Plot and optionally save a nicely formatted confusion matrix."""
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 14},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.close(fig)


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot training / validation curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy / F1
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_f1"],    label="Val F1",    marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy / F1 Curves"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to {save_path}")

    plt.close(fig)
