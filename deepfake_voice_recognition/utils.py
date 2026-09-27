"""
utils.py
--------
Shared utilities for reproducibility, checkpointing, configuration export,
and visualization of learning curves and confusion matrices.
"""

import json
import os
import random
from typing import Any, Dict, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from .config import cfg
except ImportError:
    from config import cfg


# ------------------------------------------------------------------ #
# Reproducibility
# ------------------------------------------------------------------ #
def set_seed(seed: Optional[int] = None) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    seed = cfg.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ #
# Checkpointing
# ------------------------------------------------------------------ #
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_acc: float,
    epochs_no_improve: int,
    history: Dict[str, Any],
) -> None:
    """Saves model weights, optimizer/scaler state, metrics, and writes companion config.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_acc": best_val_acc,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
        },
        path,
    )

    # Save metadata alongside checkpoint
    config_data = {
        "model_name": cfg.model_name,
        "num_unfrozen_layers": cfg.num_unfrozen_layers,
        "sample_rate": cfg.sample_rate,
        "max_seconds": cfg.max_seconds,
        "num_samples": cfg.num_samples,
        "genuine_label": cfg.genuine_label,
        "spoof_label": cfg.spoof_label,
        "best_val_acc": best_val_acc,
        "epoch": epoch,
        "labels": {"0": "genuine", "1": "spoof"},
    }
    config_path = os.path.join(os.path.dirname(path), "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)


def load_checkpoint(path: str, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    """Loads a PyTorch checkpoint dictionary."""
    map_location = map_location or cfg.device
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at target path: {path}")
    return torch.load(path, map_location=map_location)


def load_model_for_inference(
    model_cls: Type[nn.Module],
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Initializes model, restores checkpoint weights, and sets eval mode."""
    device = device or cfg.device
    checkpoint_path = checkpoint_path or cfg.ckpt_best
    ckpt = load_checkpoint(checkpoint_path, map_location=device)

    model = model_cls().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "N/A")
    val_acc = ckpt.get("best_val_acc", float("nan"))
    print(f"Loaded checkpoint '{checkpoint_path}' (Epoch {epoch}, Best Val Acc: {val_acc:.2f}%)")
    return model, ckpt


# ------------------------------------------------------------------ #
# Visualization
# ------------------------------------------------------------------ #
def plot_history(history: Dict[str, list], save_path: Optional[str] = None) -> None:
    """Plots training and validation loss & accuracy curves from history dictionary."""
    save_path = save_path or cfg.plot_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    axes[0].plot(epochs_range, history["train_loss"], marker="o", markersize=4, label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs_range, history["val_loss"], marker="s", markersize=4, label="Val Loss", color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs_range, history["train_acc"], marker="o", markersize=4, label="Train Acc", color="#2ca02c")
    axes[1].plot(epochs_range, history["val_acc"], marker="s", markersize=4, label="Val Acc", color="#d62728")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training plot to: {save_path}")
    plt.show()


def plot_training_curves_from_csv(log_path: Optional[str] = None, save_path: Optional[str] = None) -> pd.DataFrame:
    """Reads training metrics from CSV log and renders annotated learning plots."""
    log_path = log_path or cfg.log_csv_path
    save_path = save_path or cfg.plot_path

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Training log CSV not found at: {log_path}")

    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", markersize=3, label="Train Loss", alpha=0.8)
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", markersize=3, label="Val Loss", alpha=0.8)
    best_loss_idx = df["val_loss"].idxmin()
    best_loss_epoch = df.loc[best_loss_idx, "epoch"]
    best_loss_val = df.loc[best_loss_idx, "val_loss"]
    axes[0].axvline(x=best_loss_epoch, color="red", linestyle="--", alpha=0.6, label=f"Best Loss Ep ({int(best_loss_epoch)})")
    axes[0].scatter(best_loss_epoch, best_loss_val, color="red", s=50, zorder=5)
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Epoch")

    # Accuracy
    axes[1].plot(df["epoch"], df["train_acc"], marker="o", markersize=3, label="Train Acc", alpha=0.8)
    axes[1].plot(df["epoch"], df["val_acc"], marker="o", markersize=3, label="Val Acc", alpha=0.8)
    best_acc_idx = df["val_acc"].idxmax()
    best_acc_epoch = df.loc[best_acc_idx, "epoch"]
    best_acc_val = df.loc[best_acc_idx, "val_acc"]
    axes[1].axvline(x=best_acc_epoch, color="green", linestyle="--", alpha=0.6, label=f"Best Acc Ep ({int(best_acc_epoch)})")
    axes[1].scatter(best_acc_epoch, best_acc_val, color="green", s=50, zorder=5)
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved CSV curve plot to: {save_path}")
    plt.show()
    return df


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Tuple[str, ...] = ("Genuine", "Spoof"),
    save_path: Optional[str] = None,
) -> None:
    """Renders a confusion matrix heatmap."""
    import seaborn as sns

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Test Set Confusion Matrix", fontsize=13, pad=12)
    plt.ylabel("Ground Truth Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix heatmap to {save_path}")
    plt.show()