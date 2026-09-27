"""
main.py
-------
Unified Command Line Interface (CLI) for deepfake voice recognition.

Commands:
    segment   - Preprocess and split raw audio files into uniform chunks
    train     - Train (or resume) the Wav2Vec2 + 1D ResNet detector
    evaluate  - Compute classification metrics & confusion matrix on test split
    predict   - Run inference on a single audio clip to get a risk score
    export    - Export trained PyTorch weights to FP32 / quantized INT8 ONNX

Usage examples:
    python -m deepfake_voice_recognition.main segment --source-root ./data/raw --output-root ./data/segmented
    python -m deepfake_voice_recognition.main train --epochs 30 --batch-size 16
    python -m deepfake_voice_recognition.main evaluate --checkpoint checkpoints/checkpoint_best.pt
    python -m deepfake_voice_recognition.main predict --audio sample.wav
    python -m deepfake_voice_recognition.main export --checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
import os
import sys

# Support execution both as package and as direct script
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import cfg
else:
    from .config import cfg


def _apply_common_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "dataset_root", None):
        cfg.dataset_root = args.dataset_root
    if getattr(args, "checkpoint_dir", None):
        cfg.checkpoint_dir = args.checkpoint_dir
    if getattr(args, "batch_size", None):
        cfg.batch_size = args.batch_size


def cmd_segment(args: argparse.Namespace) -> None:
    if __package__ is None or __package__ == "":
        from segmenter import AudioSegmenter
    else:
        from .segmenter import AudioSegmenter

    segmenter = AudioSegmenter(
        segment_seconds=args.segment_seconds,
        target_sr=args.sample_rate,
    )
    segmenter.process_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
    )


def cmd_train(args: argparse.Namespace):
    _apply_common_overrides(args)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr

    if __package__ is None or __package__ == "":
        from train import run_training
        from utils import plot_history
    else:
        from .train import run_training
        from .utils import plot_history

    model, history = run_training()
    plot_history(history)
    return model, history


def cmd_evaluate(args: argparse.Namespace) -> None:
    _apply_common_overrides(args)

    if __package__ is None or __package__ == "":
        from model import Wav2Vec2ResNetDetector
        from dataset import get_test_dataloader
        from utils import load_model_for_inference
        from evaluate import run_confusion_matrix
    else:
        from .model import Wav2Vec2ResNetDetector
        from .dataset import get_test_dataloader
        from .utils import load_model_for_inference
        from .evaluate import run_confusion_matrix

    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device=cfg.device)
    test_loader = get_test_dataloader(cfg.dataset_root, batch_size=cfg.batch_size, test_split=args.test_split)
    run_confusion_matrix(model, test_loader, device=cfg.device)


def cmd_predict(args: argparse.Namespace) -> None:
    if __package__ is None or __package__ == "":
        from model import Wav2Vec2ResNetDetector
        from utils import load_model_for_inference
        from predict import get_risk_score
    else:
        from .model import Wav2Vec2ResNetDetector
        from .utils import load_model_for_inference
        from .predict import get_risk_score

    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device=cfg.device)
    score, label = get_risk_score(model, cfg.device, args.audio, verbose=args.verbose)
    print("\n" + "=" * 40)
    print(f"Deepfake Risk Score: {score}%")
    print(f"Verdict            : {label.upper()}")
    print("=" * 40)


def cmd_export(args: argparse.Namespace) -> None:
    import torch

    if __package__ is None or __package__ == "":
        from model import Wav2Vec2ResNetDetector
        from utils import load_model_for_inference
        from export_onnx import export_to_onnx, quantize_onnx, run_onnx_inference, _load_and_fix_length
    else:
        from .model import Wav2Vec2ResNetDetector
        from .utils import load_model_for_inference
        from .export_onnx import export_to_onnx, quantize_onnx, run_onnx_inference, _load_and_fix_length

    cfg.ensure_dirs()
    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device="cpu")

    if args.audio:
        sample_waveform = _load_and_fix_length(args.audio)
    else:
        sample_waveform = torch.zeros(cfg.num_samples)

    _, dummy_input_values = export_to_onnx(model, sample_waveform)
    if not args.skip_quantize:
        quantized_path = quantize_onnx()
        outputs = run_onnx_inference(quantized_path, dummy_input_values)
        print("ONNX Sample Verification Output:", outputs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfake_voice_recognition",
        description="End-to-End Wav2Vec2 + 1D ResNet Deepfake Voice Recognition CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- segment ----
    p_seg = subparsers.add_parser("segment", help="Segment raw audio dataset into standardized fixed-length chunks.")
    p_seg.add_argument("--source-root", default=cfg.raw_dataset_root, help="Source directory with raw audio files.")
    p_seg.add_argument("--output-root", default=cfg.dataset_root, help="Output directory for segmented dataset.")
    p_seg.add_argument("--segment-seconds", type=float, default=cfg.max_seconds, help="Chunk duration in seconds.")
    p_seg.add_argument("--sample-rate", type=int, default=cfg.sample_rate, help="Target sample rate in Hz.")
    p_seg.set_defaults(func=cmd_segment)

    # ---- train ----
    p_train = subparsers.add_parser("train", help="Train (or resume training) the model.")
    p_train.add_argument("--dataset-root", default=None, help="Root folder of segmented dataset.")
    p_train.add_argument("--checkpoint-dir", default=None, help="Folder to save checkpoints.")
    p_train.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    p_train.add_argument("--epochs", type=int, default=None, help="Total epochs.")
    p_train.add_argument("--lr", type=float, default=None, help="Learning rate.")
    p_train.set_defaults(func=cmd_train)

    # ---- evaluate ----
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on the held-out test split.")
    p_eval.add_argument("--checkpoint", default=cfg.ckpt_best, help="Path to .pt checkpoint file.")
    p_eval.add_argument("--dataset-root", default=None, help="Root folder of dataset.")
    p_eval.add_argument("--checkpoint-dir", default=None, help="Directory containing checkpoints.")
    p_eval.add_argument("--batch-size", type=int, default=None, help="Batch size for evaluation.")
    p_eval.add_argument("--test-split", type=float, default=cfg.test_split, help="Test split proportion.")
    p_eval.set_defaults(func=cmd_evaluate)

    # ---- predict ----
    p_pred = subparsers.add_parser("predict", help="Get a deepfake risk score for a single audio file.")
    p_pred.add_argument("--audio", required=True, help="Path to input audio file (.wav, .mp3, etc.).")
    p_pred.add_argument("--checkpoint", default=cfg.ckpt_best, help="Path to trained checkpoint.")
    p_pred.add_argument("--verbose", action="store_true", help="Display raw logits and class probabilities.")
    p_pred.set_defaults(func=cmd_predict)

    # ---- export ----
    p_exp = subparsers.add_parser("export", help="Export PyTorch model to ONNX FP32 and INT8 quantized formats.")
    p_exp.add_argument("--checkpoint", default=cfg.ckpt_best, help="Checkpoint to export.")
    p_exp.add_argument("--audio", default=None, help="Sample audio file used to trace the graph.")
    p_exp.add_argument("--skip-quantize", action="store_true", help="Skip INT8 dynamic quantization.")
    p_exp.set_defaults(func=cmd_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()