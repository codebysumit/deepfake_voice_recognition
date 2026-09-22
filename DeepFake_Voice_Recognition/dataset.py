"""
dataset.py
----------
Dataset scanning, the PyTorch Dataset class, and dataloader builders for
the real/fake voice-clone audio dataset.

Expected folder layout under `cfg.dataset_root`:

    dataset_root/
        real/<lang>/*.wav
        fake/<lang>/*.wav
"""

import glob
import os
import random

import numpy as np
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import cfg


class VoiceDataset(Dataset):
    """
    Loads audio files from disk, resamples to `cfg.sample_rate`, and
    pads/crops every clip to a fixed length so batches can be stacked.
    """

    def __init__(self, split_files):
        self.files = split_files

    def __len__(self):
        return len(self.files)

    def _load_and_fix_length(self, path):
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

    def __getitem__(self, idx):
        path, label = self.files[idx]
        waveform = self._load_and_fix_length(path)
        return waveform, label


def scan_dataset(dataset_root=None):
    """Walks dataset_root/real and dataset_root/fake, returns list of (filepath, label)."""
    dataset_root = dataset_root or cfg.dataset_root
    files = []
    real_root = os.path.join(dataset_root, "real")
    fake_root = os.path.join(dataset_root, "fake")

    for root, label in [(real_root, cfg.genuine_label), (fake_root, cfg.spoof_label)]:
        for ext in cfg.audio_extensions:
            for path in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True):
                files.append((path, label))

    random.shuffle(files)
    return files


def build_dataloaders(dataset_root=None, batch_size=None, val_split=None):
    """Returns (train_loader, val_loader) built from a fresh shuffle+split."""
    dataset_root = dataset_root or cfg.dataset_root
    batch_size = batch_size or cfg.batch_size
    val_split = cfg.val_split if val_split is None else val_split

    all_files = scan_dataset(dataset_root)
    if len(all_files) == 0:
        raise RuntimeError(f"No audio files found under {dataset_root}. Check folder structure.")

    n_real = sum(1 for _, l in all_files if l == cfg.genuine_label)
    n_fake = sum(1 for _, l in all_files if l == cfg.spoof_label)
    print(f"Found {n_real} real files, {n_fake} fake files.")
    if n_real == 0 or n_fake == 0 or max(n_real, n_fake) / max(1, min(n_real, n_fake)) > 3:
        print("WARNING: classes look imbalanced. Consider balancing before training.")

    val_count = max(1, int(len(all_files) * val_split))
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    train_ds = VoiceDataset(train_files)
    val_ds = VoiceDataset(val_files)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
    return train_loader, val_loader


def get_test_dataloader(dataset_root=None, batch_size=None, test_split=None):
    """Returns a held-out test DataLoader, stratified by label."""
    dataset_root = dataset_root or cfg.dataset_root
    batch_size = batch_size or cfg.batch_size
    test_split = cfg.test_split if test_split is None else test_split

    all_files = scan_dataset(dataset_root)
    if len(all_files) == 0:
        raise RuntimeError(f"No audio files found under {dataset_root}. Check folder structure.")

    _, test_files = train_test_split(
        all_files, test_size=test_split, random_state=cfg.seed,
        stratify=[f[1] for f in all_files],
    )

    print(f"Total files: {len(all_files)}")
    print(f"Test files: {len(test_files)}")

    test_ds = VoiceDataset(test_files)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)
    return test_loader


def visualize_samples(dataset, num_samples=3, save_path=None):
    """Plots a few random waveforms from `dataset` for a quick sanity check."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    indices = random.sample(range(len(dataset)), num_samples)
    for i, idx in enumerate(indices):
        waveform, label = dataset[idx]
        label_str = "Genuine" if label == cfg.genuine_label else "Spoof"
        axes[i].plot(waveform)
        axes[i].set_title(f"Sample {idx} - Label: {label_str}")
        axes[i].set_xlabel("Samples")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved sample plot to {save_path}")
    plt.show()