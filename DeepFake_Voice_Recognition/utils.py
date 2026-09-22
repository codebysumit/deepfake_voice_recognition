"""
utils.py
--------
Shared helpers used by train.py, evaluate.py and predict.py:
checkpoint save/load, training-curve plotting, confusion-matrix plotting,
and seeding.
"""

import os
import random

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from config import cfg


# ------------------------------------------------------------------ #
# Reproducibility
# ------------------------------------------------------------------ #
def set_seed(seed=None):
    seed = cfg.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import json

# ------------------------------------------------------------------ #
# Checkpointing
# ------------------------------------------------------------------ #
def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_acc, epochs_no_improve, history):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
    }, path)

    # Dynamically write config.json alongside the checkpoint file
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


def load_checkpoint(path, map_location=None):
    map_location = map_location or cfg.device
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    return torch.load(path, map_location=map_location)


def load_model_for_inference(model_cls, checkpoint_path=None, device=None):
    """Builds a fresh model, loads weights from a checkpoint, sets eval mode."""
    device = device or cfg.device
    checkpoint_path = checkpoint_path or cfg.ckpt_best
    ckpt = load_checkpoint(checkpoint_path, map_location=device)

    model = model_cls().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint '{checkpoint_path}' "
          f"(epoch {ckpt.get('epoch')}, best_val_acc={ckpt.get('best_val_acc', float('nan')):.2f}%)")
    return model, ckpt


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #
def plot_history(history, save_path=None):
    """Plots loss/accuracy curves from an in-memory `history` dict."""
    save_path = save_path or cfg.plot_path
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs_range, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["train_acc"], marker="o", label="Train Acc")
    axes[1].plot(epochs_range, history["val_acc"], marker="o", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved plot to {save_path}")


def plot_training_curves_from_csv(log_path=None, save_path=None):
    """Same as plot_history but reads from the CSV log (survives a fresh process)."""
    log_path = log_path or cfg.log_csv_path
    save_path = save_path or cfg.plot_path

    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", markersize=3, label="Train Loss", alpha=0.7)
    axes[0].plot(df["epoch"], df["val_loss"], marker="o", markersize=3, label="Val Loss", alpha=0.7)
    best_loss_idx = df["val_loss"].idxmin()
    best_loss_epoch = df.loc[best_loss_idx, "epoch"]
    best_loss_val = df.loc[best_loss_idx, "val_loss"]
    axes[0].axvline(x=best_loss_epoch, color="red", linestyle="--", alpha=0.6,
                     label=f"Best Epoch ({int(best_loss_epoch)})")
    axes[0].scatter(best_loss_epoch, best_loss_val, color="red", s=50, zorder=5)
    axes[0].annotate(f"Best Loss:\n{best_loss_val:.4f}", (best_loss_epoch, best_loss_val),
                      textcoords="offset points", xytext=(0, 15), ha="center", color="red", weight="bold")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Epoch")

    # Accuracy
    axes[1].plot(df["epoch"], df["train_acc"], marker="o", markersize=3, label="Train Acc", alpha=0.7)
    axes[1].plot(df["epoch"], df["val_acc"], marker="o", markersize=3, label="Val Acc", alpha=0.7)
    best_acc_idx = df["val_acc"].idxmax()
    best_acc_epoch = df.loc[best_acc_idx, "epoch"]
    best_acc_val = df.loc[best_acc_idx, "val_acc"]
    axes[1].axvline(x=best_acc_epoch, color="green", linestyle="--", alpha=0.6,
                     label=f"Best Epoch ({int(best_acc_epoch)})")
    axes[1].scatter(best_acc_epoch, best_acc_val, color="green", s=50, zorder=5)
    axes[1].annotate(f"Best Acc:\n{best_acc_val:.2f}%", (best_acc_epoch, best_acc_val),
                      textcoords="offset points", xytext=(0, 15), ha="center", color="green", weight="bold")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved plot to {save_path}")
    return df


def plot_confusion_matrix(cm, class_names=("Genuine", "Spoof"), save_path=None):
    """Plots a confusion matrix given a precomputed `cm` array (see evaluate.py)."""
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()