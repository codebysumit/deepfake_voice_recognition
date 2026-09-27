"""
export_onnx.py
--------------
Exports the trained PyTorch Wav2Vec2 + 1D ResNet model to ONNX format
and creates a dynamically quantized (INT8) model optimized for fast CPU inference.

Usage:
    python -m deepfake_voice_recognition.export_onnx --checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

try:
    from .config import cfg
    from .model import Wav2Vec2ResNetDetector
    from .utils import load_model_for_inference
except ImportError:
    from config import cfg
    from model import Wav2Vec2ResNetDetector
    from utils import load_model_for_inference


def _load_and_fix_length(audio_path: str) -> torch.Tensor:
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
    return torch.tensor(waveform.astype(np.float32))


def export_to_onnx(
    model: Wav2Vec2ResNetDetector,
    sample_waveform: torch.Tensor,
    output_path: Optional[str] = None,
) -> Tuple[str, torch.Tensor]:
    """Exports model to ONNX with dynamic batch and time dimensions."""
    output_path = output_path or cfg.onnx_fp32_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = model.to("cpu").eval()
    dummy_input_values = model.preprocess([sample_waveform]).to("cpu")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input_values,
            output_path,
            input_names=["input_values"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            dynamic_axes={
                "input_values": {0: "batch", 1: "time"},
                "output": {0: "batch"},
            },
            dynamo=False,
        )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully exported FP32 ONNX model to: {output_path} ({file_size_mb:.2f} MB)")
    return output_path, dummy_input_values


def quantize_onnx(
    onnx_path: Optional[str] = None,
    quantized_path: Optional[str] = None,
) -> str:
    """Dynamic INT8 quantization targeting MatMul operations for optimal CPU throughput."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = onnx_path or cfg.onnx_fp32_path
    quantized_path = quantized_path or cfg.onnx_quantized_path

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    file_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"Quantized INT8 ONNX model saved to: {quantized_path} ({file_size_mb:.2f} MB)")
    return quantized_path


def run_onnx_inference(quantized_path: str, input_values: torch.Tensor):
    """Executes test inference using ONNX Runtime."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(quantized_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input_values": input_values.cpu().numpy()})
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Export and quantize the model to ONNX.")
    parser.add_argument("--checkpoint", default=cfg.ckpt_best, help="Path to .pt checkpoint.")
    parser.add_argument("--audio", default=None, help="Sample audio file used to trace the export.")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip INT8 quantization.")
    args = parser.parse_args()

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
        print("ONNX Runtime Verification Output:", outputs)


if __name__ == "__main__":
    main()