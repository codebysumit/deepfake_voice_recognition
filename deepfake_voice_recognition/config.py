"""
config.py
---------
Centralized configuration management for paths, hyperparameters, and audio
processing settings across the deepfake voice recognition project.

Usage:
    from deepfake_voice_recognition.config import cfg
    print(cfg.batch_size)
    print(cfg.num_samples)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

try:
    import torch
    _DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    _DEFAULT_DEVICE = "cpu"


@dataclass
class Config:
    # ---- Model Architecture ----
    model_name: str = "facebook/wav2vec2-base"    # e.g., "facebook/wav2vec2-large-xlsr-53" for multilingual/Indic
    num_unfrozen_layers: int = 4                   # last N transformer encoder layers to fine-tune

    # ---- Audio Preprocessing & Features ----
    sample_rate: int = 16000
    max_seconds: float = 4.0
    audio_extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
    genuine_label: int = 0
    spoof_label: int = 1

    # ---- Data Directories ----
    raw_dataset_root: str = "./data/raw"
    dataset_root: str = "./data/indian-deepfake-voice"   # expects <root>/real/** and <root>/fake/**
    val_split: float = 0.15
    test_split: float = 0.20

    # ---- Training Hyperparameters ----
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 1e-4
    patience: int = 20
    num_workers: int = 2
    seed: int = 42

    # ---- Checkpoints / Artifacts / Exports ----
    checkpoint_dir: str = "./checkpoints"
    onnx_dir: str = "./exported"

    # ---- Hardware Device ----
    device: Any = field(default_factory=lambda: _DEFAULT_DEVICE)

    @property
    def num_samples(self) -> int:
        """Total discrete audio samples per input chunk."""
        return int(self.sample_rate * self.max_seconds)

    @property
    def ckpt_latest(self) -> str:
        return os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")

    @property
    def ckpt_best(self) -> str:
        return os.path.join(self.checkpoint_dir, "checkpoint_best.pt")

    @property
    def plot_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "training_curves.png")

    @property
    def log_csv_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "training_log.csv")

    @property
    def onnx_fp32_path(self) -> str:
        return os.path.join(self.onnx_dir, "model.onnx")

    @property
    def onnx_quantized_path(self) -> str:
        return os.path.join(self.onnx_dir, "model_quantized.onnx")

    def ensure_dirs(self) -> None:
        """Ensures all required output directories exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.dataset_root) if os.path.dirname(self.dataset_root) else ".", exist_ok=True)


# Global singleton configuration instance
cfg = Config()