"""
predict.py
----------
Run inference on a single audio file and get a "risk score" (probability
the clip is AI-generated / spoofed).

Run directly:
    python predict.py --audio path/to/clip.wav
    python predict.py --audio path/to/clip.wav --checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
import os
import sys

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F

from config import cfg
from model import Wav2Vec2ResNetDetector
from utils import load_model_for_inference


def _load_and_fix_length(audio_path):
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


def get_risk_score(model, device, audio_path, verbose=False):
    """
    Returns (risk_score 0-100, label) where risk_score is the probability
    the audio is AI-generated/spoofed.
    """
    waveform = _load_and_fix_length(audio_path)
    device = next(model.parameters()).device
    waveform_tensor = torch.tensor(waveform)

    model.eval()
    with torch.no_grad():
        input_values = model.preprocess([waveform_tensor]).to(device)
        logits = model(input_values)
        probs = F.softmax(logits, dim=1)[0]

    if verbose:
        print(f"Logits: {logits}")
        print(f"Probabilities: {probs}")

    spoof_prob = probs[cfg.spoof_label].item()
    risk_score = round(spoof_prob * 100, 2)
    label = "spoof" if spoof_prob >= 0.5 else "genuine"
    return risk_score, label


def main():
    parser = argparse.ArgumentParser(description="Get a deepfake risk score for an audio file.")
    parser.add_argument("--audio", required=True, help="Path to a .wav/.flac/.mp3 file.")
    parser.add_argument("--checkpoint", default=cfg.ckpt_best, help="Path to a .pt checkpoint.")
    parser.add_argument("--verbose", action="store_true", help="Print raw logits/probabilities.")
    args = parser.parse_args()

    device = cfg.device
    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device=device)

    score, label = get_risk_score(model, device, args.audio, verbose=args.verbose)
    print(f"Risk score: {score}% -> {label}")


if __name__ == "__main__":
    main()