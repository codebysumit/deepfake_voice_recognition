"""
predict.py
----------
Inference module: computes a deepfake probability ("risk score" 0-100%)
for a single audio file (.wav, .mp3, .flac, etc.).

Usage:
    python -m deepfake_voice_recognition.predict --audio sample.wav
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

try:
    from .config import cfg
    from .model import Wav2Vec2ResNetDetector
    from .utils import load_model_for_inference
except ImportError:
    from config import cfg
    from model import Wav2Vec2ResNetDetector
    from utils import load_model_for_inference


def load_and_standardize_waveform(audio_path: str) -> np.ndarray:
    """Loads audio from disk, downmixes to mono, resamples to cfg.sample_rate, and standardizes length."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, orig_sr = sf.read(audio_path, always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if orig_sr != cfg.sample_rate:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=cfg.sample_rate)

    num_samples = cfg.num_samples
    if len(waveform) >= num_samples:
        waveform = waveform[:num_samples]
    else:
        num_repeats = int(np.ceil(num_samples / len(waveform)))
        waveform = np.tile(waveform, num_repeats)[:num_samples]

    return waveform.astype(np.float32)


def get_risk_score(
    model: Wav2Vec2ResNetDetector,
    device: Optional[torch.device] = None,
    audio_path: str = "",
    verbose: bool = False,
) -> Tuple[float, str]:
    """
    Computes the deepfake risk score for the specified audio clip.

    Returns:
        (risk_score_percentage, classification_label)
    """
    device = device or next(model.parameters()).device
    waveform = load_and_standardize_waveform(audio_path)
    waveform_tensor = torch.from_numpy(waveform)

    model.eval()
    with torch.no_grad():
        input_values = model.preprocess([waveform_tensor]).to(device)
        logits = model(input_values)
        probs = F.softmax(logits, dim=1)[0]

    if verbose:
        print(f"[Debug] Logits: {logits.cpu().numpy().tolist()}")
        print(f"[Debug] Probabilities (Genuine, Spoof): {probs.cpu().numpy().tolist()}")

    spoof_prob = probs[cfg.spoof_label].item()
    risk_score = round(spoof_prob * 100, 2)
    label = "spoof" if spoof_prob >= 0.5 else "genuine"
    return risk_score, label


def main():
    parser = argparse.ArgumentParser(description="Estimate deepfake risk score for an audio file.")
    parser.add_argument("--audio", required=True, help="Path to .wav / .mp3 / .flac audio file.")
    parser.add_argument("--checkpoint", default=cfg.ckpt_best, help="Path to .pt checkpoint file.")
    parser.add_argument("--verbose", action="store_true", help="Print raw network activations.")
    args = parser.parse_args()

    device = cfg.device
    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device=device)
    score, label = get_risk_score(model, device, args.audio, verbose=args.verbose)
    print(f"Risk score: {score}% -> {label.upper()}")


if __name__ == "__main__":
    main()