"""
dataset.py
----------
Dataset scanning, PyTorch Dataset abstraction, and balanced DataLoader
builders for deepfake audio classification.

Expected folder structure under `dataset_root`:
    dataset_root/
        real/<lang>/*.wav
        fake/<lang>/*.wav
"""

import glob
import os
import random
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from .config import cfg
except ImportError:
    from config import cfg


class VoiceDataset(Dataset):
    """
    Loads raw audio waveforms from disk, resamples to `cfg.sample_rate`, and
    pads/crops every clip to a fixed length (`cfg.num_samples`) so batches can be stacked.
    """

    def __init__(self, split_files: List[Tuple[str, int]]):
        self.files = split_files

    def __len__(self) -> int:
        return len(self.files)

    def _load_and_fix_length(self, path: str) -> np.ndarray:
        waveform, orig_sr = sf.read(path, always_2d=False)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        if orig_sr != cfg.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=cfg.sample_rate)

        num_samples = cfg.num_samples
        if len(waveform) >= num_samples:
            start = random.randint(0, len(waveform) - num_samples)
            waveform = waveform[start:start + num_samples]
        else:
            num_repeats = int(np.ceil(num_samples / len(waveform)))
            waveform = np.tile(waveform, num_repeats)[:num_samples]

        return waveform.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        path, label = self.files[idx]
        waveform = self._load_and_fix_length(path)
        return waveform, label


def scan_dataset(dataset_root: Optional[str] = None) -> List[Tuple[str, int]]:
    """Walks dataset_root/real and dataset_root/fake, returns shuffled list of (filepath, label)."""
    dataset_root = dataset_root or cfg.dataset_root
    files = []
    real_root = os.path.join(dataset_root, "real")
    fake_root = os.path.join(dataset_root, "fake")

    for root, label in [(real_root, cfg.genuine_label), (fake_root, cfg.spoof_label)]:
        if not os.path.exists(root):
            continue
        for ext in cfg.audio_extensions:
            for path in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True):
                files.append((path, label))

    random.shuffle(files)
    return files


def build_dataloaders(
    dataset_root: Optional[str] = None,
    batch_size: Optional[int] = None,
    val_split: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) built from a fresh shuffle and split."""
    dataset_root = dataset_root or cfg.dataset_root
    batch_size = batch_size or cfg.batch_size
    val_split = cfg.val_split if val_split is None else val_split

    all_files = scan_dataset(dataset_root)
    if len(all_files) == 0:
        raise RuntimeError(f"No audio files found under {dataset_root}. Check directory structure.")

    n_real = sum(1 for _, l in all_files if l == cfg.genuine_label)
    n_fake = sum(1 for _, l in all_files if l == cfg.spoof_label)
    print(f"Dataset scan: {n_real} genuine (real) samples, {n_fake} spoof (fake) samples.")
    if n_real == 0 or n_fake == 0 or max(n_real, n_fake) / max(1, min(n_real, n_fake)) > 3:
        print("[Warning] Class imbalance detected. Consider dataset rebalancing.")

    val_count = max(1, int(len(all_files) * val_split))
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    train_ds = VoiceDataset(train_files)
    val_ds = VoiceDataset(val_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")
    return train_loader, val_loader


def get_test_dataloader(
    dataset_root: Optional[str] = None,
    batch_size: Optional[int] = None,
    test_split: Optional[float] = None,
) -> DataLoader:
    """Returns a held-out test DataLoader, stratified by ground-truth label."""
    dataset_root = dataset_root or cfg.dataset_root
    batch_size = batch_size or cfg.batch_size
    test_split = cfg.test_split if test_split is None else test_split

    all_files = scan_dataset(dataset_root)
    if len(all_files) == 0:
        raise RuntimeError(f"No audio files found under {dataset_root}. Check folder structure.")

    _, test_files = train_test_split(
        all_files,
        test_size=test_split,
        random_state=cfg.seed,
        stratify=[f[1] for f in all_files],
    )

    print(f"Total dataset: {len(all_files)} files | Held-out test set: {len(test_files)} files")

    test_ds = VoiceDataset(test_files)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return test_loader


def visualize_samples(dataset: Dataset, num_samples: int = 3, save_path: Optional[str] = None) -> None:
    """Plots random waveforms from `dataset` for sanity verification."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3.5 * num_samples))
    if num_samples == 1:
        axes = [axes]

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for i, idx in enumerate(indices):
        waveform, label = dataset[idx]
        label_str = "Genuine" if label == cfg.genuine_label else "Spoof"
        axes[i].plot(waveform, color="#1f77b4" if label == cfg.genuine_label else "#d62728", alpha=0.85)
        axes[i].set_title(f"Sample #{idx} — Label: {label_str}")
        axes[i].set_xlabel("Time (Samples)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved waveform plot to {save_path}")
    plt.show()