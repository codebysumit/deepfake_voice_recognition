"""
config.py
---------
Central place for every path, hyperparameter and constant used across the
project. Import this module instead of hard-coding values in other files:

    from config import cfg
    print(cfg.batch_size)

Edit the values below (or override them with environment variables / CLI
flags in main.py) to control a run.
"""

import os
from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    # ---- Model ----
    model_name: str = "facebook/wav2vec2-base"   # swap to "facebook/wav2vec2-large-xlsr-53" for Indic robustness
    num_unfrozen_layers: int = 4                  # last N transformer layers fine-tuned, rest frozen

    # ---- Audio ----
    sample_rate: int = 16000
    max_seconds: int = 4
    audio_extensions: tuple = (".wav", ".flac", ".mp3")
    genuine_label: int = 0
    spoof_label: int = 1

    # ---- Data ----
    dataset_root: str = "./data/indian-deepfake-voice"   # expects <root>/real/** and <root>/fake/**
    val_split: float = 0.15
    test_split: float = 0.2

    # ---- Training ----
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 1e-4
    patience: int = 20
    num_workers: int = 2
    seed: int = 42

    # ---- Checkpoints / logs / outputs ----
    checkpoint_dir: str = "./checkpoints"
    onnx_dir: str = "./exported"

    # ---- Device ----
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    @property
    def num_samples(self) -> int:
        return self.sample_rate * self.max_seconds

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

    def ensure_dirs(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)


# Single shared instance every module imports. Mutate this (or reassign
# fields) before calling into train/evaluate/predict if you need a
# different setup for a given run.
cfg = Config()