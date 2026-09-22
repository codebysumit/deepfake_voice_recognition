"""
train.py
--------
Resumable, mixed-precision training loop for the Wav2Vec2ResNetDetector.

Run directly:
    python train.py

Or import as a module:
    from train import run_training
    model, history = run_training()

Safe to re-run after an interruption (e.g. a Colab disconnect) — it
automatically resumes from `cfg.ckpt_latest` if that file exists.
"""

import csv
import gc
import os
import sys

# Ensure current directory is in sys.path so this runs both as a script and as a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm

from config import cfg
from dataset import build_dataloaders
from model import Wav2Vec2ResNetDetector
from utils import save_checkpoint, plot_history, set_seed


def run_training():
    cfg.ensure_dirs()
    set_seed()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = cfg.device
    print(f"Using device: {device}")
    print(f"Starting training from dataset_root: {cfg.dataset_root}")

    train_loader, val_loader = build_dataloaders(cfg.dataset_root, batch_size=cfg.batch_size)

    model = Wav2Vec2ResNetDetector(num_unfrozen_layers=cfg.num_unfrozen_layers).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 1
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    csv_headers = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]

    if os.path.exists(cfg.ckpt_latest):
        ckpt = torch.load(cfg.ckpt_latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        history = ckpt["history"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, cfg.epochs + 1), desc="Epochs"):
        # ---- Train ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for waveforms, labels in pbar:
            waveforms = model.preprocess(waveforms).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(waveforms)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                               "acc": f"{100.0 * train_correct / train_total:.2f}%"})

        t_loss = train_loss / train_total
        t_acc = 100.0 * train_correct / train_total

        # ---- Validate ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False)
        with torch.no_grad():
            for waveforms, labels in pbar_val:
                waveforms = model.preprocess(waveforms).to(device)
                labels = labels.to(device)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    logits = model(waveforms)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
                pbar_val.set_postfix({"v_loss": f"{loss.item():.4f}",
                                       "v_acc": f"{100.0 * val_correct / val_total:.2f}%"})

        v_loss = val_loss / val_total
        v_acc = 100.0 * val_correct / val_total
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        # ---- Log to CSV ----
        file_exists = os.path.isfile(cfg.log_csv_path)
        with open(cfg.log_csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "train_loss": f"{t_loss:.6f}", "train_acc": f"{t_acc:.2f}",
                              "val_loss": f"{v_loss:.6f}", "val_acc": f"{v_acc:.2f}"})

        # ---- Checkpoint ----
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            epochs_no_improve = 0
            save_checkpoint(cfg.ckpt_best, model, optimizer, scaler, epoch, best_val_acc, 0, history)
        else:
            epochs_no_improve += 1

        save_checkpoint(cfg.ckpt_latest, model, optimizer, scaler, epoch, best_val_acc, epochs_no_improve, history)

        print(f"Epoch {epoch}: train_loss={t_loss:.4f} train_acc={t_acc:.2f}% "
              f"val_loss={v_loss:.4f} val_acc={v_acc:.2f}% (best={best_val_acc:.2f}%)")

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping: no improvement in {cfg.patience} epochs.")
            break
        gc.collect()

    return model, history


if __name__ == "__main__":
    trained_model, trained_history = run_training()
    plot_history(trained_history)