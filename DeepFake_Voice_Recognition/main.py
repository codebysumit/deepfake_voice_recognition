"""
main.py
-------
Single entrypoint for the whole project. Every stage (train / evaluate /
predict / export) is also usable as its own script, but this file lets you
drive the full pipeline with one command and is the intended `python -m`
target for the project.

Usage:
    python main.py train
    python main.py train --dataset-root /path/to/data --epochs 30
    python main.py evaluate --checkpoint checkpoints/checkpoint_best.pt
    python main.py predict --audio sample.wav
    python main.py export --checkpoint checkpoints/checkpoint_best.pt --audio sample.wav
"""

import argparse
import os
import sys

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg


def _apply_common_overrides(args):
    if getattr(args, "dataset_root", None):
        cfg.dataset_root = args.dataset_root
    if getattr(args, "checkpoint_dir", None):
        cfg.checkpoint_dir = args.checkpoint_dir
    if getattr(args, "batch_size", None):
        cfg.batch_size = args.batch_size


def cmd_train(args):
    _apply_common_overrides(args)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr

    from train import run_training
    from utils import plot_history

    model, history = run_training()
    plot_history(history)
    return model, history


def cmd_evaluate(args):
    _apply_common_overrides(args)

    from model import Wav2Vec2ResNetDetector
    from dataset import get_test_dataloader
    from utils import load_model_for_inference
    from evaluate import run_confusion_matrix

    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device=cfg.device)
    test_loader = get_test_dataloader(cfg.dataset_root, batch_size=cfg.batch_size, test_split=args.test_split)
    run_confusion_matrix(model, test_loader, device=cfg.device)


def cmd_predict(args):
    from model import Wav2Vec2ResNetDetector
    from utils import load_model_for_inference
    from predict import get_risk_score

    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device=cfg.device)
    score, label = get_risk_score(model, cfg.device, args.audio, verbose=args.verbose)
    print(f"Risk score: {score}% -> {label}")


def cmd_export(args):
    from model import Wav2Vec2ResNetDetector
    from utils import load_model_for_inference
    from export_onnx import export_to_onnx, quantize_onnx, run_onnx_inference
    import torch

    cfg.ensure_dirs()
    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device="cpu")

    if args.audio:
        from export_onnx import _load_and_fix_length
        sample_waveform = _load_and_fix_length(args.audio)
    else:
        sample_waveform = torch.zeros(cfg.num_samples)

    _, dummy_input_values = export_to_onnx(model, sample_waveform)
    if not args.skip_quantize:
        quantized_path = quantize_onnx()
        outputs = run_onnx_inference(quantized_path, dummy_input_values)
        print("Sample ONNX output:", outputs)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="deepfake_voice_detector",
        description="Wav2Vec2 + 1D ResNet AI-voice / deepfake-audio detector.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = subparsers.add_parser("train", help="Train (or resume training) the model.")
    p_train.add_argument("--dataset-root", default=None)
    p_train.add_argument("--checkpoint-dir", default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.set_defaults(func=cmd_train)

    # ---- evaluate ----
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on the test split.")
    p_eval.add_argument("--checkpoint", default=cfg.ckpt_best)
    p_eval.add_argument("--dataset-root", default=None)
    p_eval.add_argument("--checkpoint-dir", default=None)
    p_eval.add_argument("--batch-size", type=int, default=None)
    p_eval.add_argument("--test-split", type=float, default=cfg.test_split)
    p_eval.set_defaults(func=cmd_evaluate)

    # ---- predict ----
    p_pred = subparsers.add_parser("predict", help="Get a risk score for a single audio file.")
    p_pred.add_argument("--audio", required=True)
    p_pred.add_argument("--checkpoint", default=cfg.ckpt_best)
    p_pred.add_argument("--verbose", action="store_true")
    p_pred.set_defaults(func=cmd_predict)

    # ---- export ----
    p_exp = subparsers.add_parser("export", help="Export to ONNX and quantize for CPU inference.")
    p_exp.add_argument("--checkpoint", default=cfg.ckpt_best)
    p_exp.add_argument("--audio", default=None, help="Sample audio used to trace the export.")
    p_exp.add_argument("--skip-quantize", action="store_true")
    p_exp.set_defaults(func=cmd_export)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()