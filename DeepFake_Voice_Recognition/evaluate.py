"""
evaluate.py
-----------
Evaluates a trained checkpoint on a held-out test split: prints a
classification report and plots a confusion matrix.

Run directly:
    python evaluate.py --checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
import os
import sys

# Ensure current directory is in sys.path so this runs both as a script and as a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from config import cfg
from dataset import get_test_dataloader
from model import Wav2Vec2ResNetDetector
from utils import load_model_for_inference, plot_confusion_matrix


def evaluate_model_performance(model, dataloader, device=None):
    """Prints precision/recall/F1 classification report. Returns (all_labels, all_preds)."""
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

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["genuine", "spoof"]))
    return all_labels, all_preds


def run_confusion_matrix(model, dataloader, device=None, save_path=None):
    """Computes and plots a confusion matrix for `model` on `dataloader`."""
    all_labels, all_preds = evaluate_model_performance(model, dataloader, device)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names=("Genuine", "Spoof"), save_path=save_path)
    return cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split.")
    parser.add_argument("--checkpoint", default=cfg.ckpt_best, help="Path to a .pt checkpoint.")
    parser.add_argument("--dataset-root", default=cfg.dataset_root, help="Root of the dataset.")
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--test-split", type=float, default=cfg.test_split)
    args = parser.parse_args()

    device = cfg.device
    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device=device)

    test_loader = get_test_dataloader(args.dataset_root, batch_size=args.batch_size, test_split=args.test_split)
    run_confusion_matrix(model, test_loader, device=device)


if __name__ == "__main__":
    main()