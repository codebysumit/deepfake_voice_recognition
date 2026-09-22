"""
export_onnx.py
---------------
Exports the trained PyTorch model to ONNX and produces a dynamically
quantized (INT8) version for fast CPU inference on lower-powered machines.

NOTE: quantization only targets MatMul ops (`op_types_to_quantize=["MatMul"]`).
Quantizing the Conv1d layers as well makes CPU inference *slower* on many
ONNX Runtime builds for this architecture — MatMul-only quantization is the
fast path that was validated for this model.

Run directly:
    python export_onnx.py --checkpoint checkpoints/checkpoint_best.pt

Then run inference on the exported model with:
    python export_onnx.py --run-sample --audio path/to/clip.wav
"""

import argparse
import os
import sys

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import soundfile as sf
import librosa

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
    return torch.tensor(waveform.astype(np.float32))


def export_to_onnx(model, sample_waveform, output_path=None):
    """Exports `model` to a static-opset ONNX file with dynamic batch/time axes."""
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
            dynamo=False,  # legacy TorchScript-based tracer — more predictable for this model
        )

    print(f"Exported ONNX model to {output_path} "
          f"({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")
    return output_path, dummy_input_values


def quantize_onnx(onnx_path=None, quantized_path=None):
    """Dynamic INT8 quantization, MatMul-only (fast path for this architecture)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_path = onnx_path or cfg.onnx_fp32_path
    quantized_path = quantized_path or cfg.onnx_quantized_path

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    print(f"Quantized ONNX model saved to {quantized_path} "
          f"({os.path.getsize(quantized_path) / (1024 * 1024):.2f} MB)")

    import onnx
    op_types = set(n.op_type for n in onnx.load(quantized_path).graph.node)
    print("Op types in quantized model:", op_types)
    if "QLinearConv" in op_types:
        print("WARNING: Conv layers were quantized — this is likely the slow variant. "
              "Re-run with op_types_to_quantize=['MatMul'] only.")
    return quantized_path


def run_onnx_inference(quantized_path, input_values):
    """Runs a single forward pass through the quantized ONNX model."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(quantized_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input_values": input_values.cpu().numpy()})
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Export and quantize the model to ONNX.")
    parser.add_argument("--checkpoint", default=cfg.ckpt_best)
    parser.add_argument("--audio", default=None, help="Sample audio file used to trace the export.")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path=args.checkpoint, device="cpu")

    if args.audio:
        sample_waveform = _load_and_fix_length(args.audio)
    else:
        # Deterministic dummy waveform when no sample audio is provided.
        sample_waveform = torch.zeros(cfg.num_samples)

    _, dummy_input_values = export_to_onnx(model, sample_waveform)

    if not args.skip_quantize:
        quantized_path = quantize_onnx()
        outputs = run_onnx_inference(quantized_path, dummy_input_values)
        print("Sample ONNX output:", outputs)


if __name__ == "__main__":
    main()