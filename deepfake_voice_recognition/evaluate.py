"""
evaluate.py
-----------
Evaluates a trained model checkpoint on the held-out test split,
producing classification metrics (precision, recall, F1, accuracy)
and an annotated confusion matrix plot.

Usage:
    python -m deepfake_voice_recognition.evaluate --checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

try:
    from .config import cfg
    from .dataset import get_test_dataloader
    from .model import Wav2Vec2ResNetDetector
    from .utils import load_model_for_inference, plot_confusion_matrix
except ImportError:
    from config import cfg
    from dataset import get_test_dataloader
    from model import Wav2Vec2ResNetDetector
    from utils import load_model_for_inference, plot_confusion_matrix


def evaluate_model_performance(
    model: Wav2Vec2ResNetDetector,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes predictions and logs full classification report."""
    device = device or cfg.device
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            waveforms = model.preprocess(waveforms).to(device)
            labels = labels.to(device)

            logits = model(waveforms)
            predictions = logits.argmax(dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    print("\n" + "=" * 50)
    print("Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["Genuine", "Spoof"], digits=4))
    return y_true, y_pred


def run_confusion_matrix(
    model: Wav2Vec2ResNetDetector,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Computes and visualizes confusion matrix."""
    all_labels, all_preds = evaluate_model_performance(model, dataloader, device)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names=("Genuine", "Spoof"), save_path=save_path)
    return cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split.")
    parser.add_argument("--checkpoint", default=cfg.ckpt_best, help="Path to .pt model checkpoint.")
    parser.add_argument("--dataset-root", default=cfg.dataset_root, help="Root folder of the dataset.")
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size, help="Evaluation batch size.")
    parser.add_argument("--test-split", type=float, default=cfg.test_split, help="Test split proportion.")
    args = parser.parse_args()

    device = cfg.device
    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device=device)
    test_loader = get_test_dataloader(args.dataset_root, batch_size=args.batch_size, test_split=args.test_split)
    run_confusion_matrix(model, test_loader, device=device)


if __name__ == "__main__":
    main()