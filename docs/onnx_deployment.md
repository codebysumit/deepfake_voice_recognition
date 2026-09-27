# ONNX Export & Edge CPU Deployment

This guide explains exporting the PyTorch model to ONNX and applying dynamic INT8 quantization for real-time edge CPU deployment.

---

## 1. Why ONNX for Deepfake Detection?

Deploying large speech Transformer backbones in production environments (e.g. call centers, edge gateways, mobile backends) requires minimizing latency and eliminating heavy GPU runtime dependencies.

- **Portability**: Run predictions across C++, Rust, Go, Python, and Node.js with ONNX Runtime.
- **Dynamic Axes**: The exported model supports dynamic `batch` and `time` dimensions.
- **Quantization**: Compresses model footprint from ~360 MB (FP32) to ~95 MB (INT8) while reducing CPU latency by 2x–3x.

---

## 2. MatMul-Only Quantization Insight

During benchmarking of Wav2Vec2 + 1D ResNet:
- Quantizing **Conv1D layers** on CPU with ONNX Runtime (`QLinearConv`) introduces overhead on small kernel sizes, leading to **slower** execution.
- Restricting quantization strictly to linear and attention projection operations (`op_types_to_quantize=["MatMul"]`) delivers maximum throughput and speedups on modern x86 / ARM CPUs.

---

## 3. Running Export & Quantization

```bash
python -m deepfake_voice_recognition.main export \
    --checkpoint checkpoints/checkpoint_best.pt
```

Generated artifacts:
- `exported/model.onnx` (FP32 model)
- `exported/model_quantized.onnx` (Dynamic INT8 model)

---

## 4. Standalone ONNX Inference Example (Python)

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa

# 1. Load and resample audio
audio_path = "test_sample.wav"
waveform, sr = sf.read(audio_path, always_2d=False)
if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)
if sr != 16000:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

# 2. Normalize and fix length (4 seconds = 64,000 samples)
target_len = 64000
if len(waveform) >= target_len:
    waveform = waveform[:target_len]
else:
    waveform = np.pad(waveform, (0, target_len - len(waveform)))

# Standardize feature amplitude
input_tensor = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-7)
input_tensor = input_tensor.astype(np.float32)[np.newaxis, :]  # Shape: (1, 64000)

# 3. Execute ONNX Runtime Session
session = ort.InferenceSession("exported/model_quantized.onnx", providers=["CPUExecutionProvider"])
outputs = session.run(None, {"input_values": input_tensor})
logits = outputs[0][0]

# Compute probabilities
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)
spoof_risk = probs[1] * 100

print(f"Spoof Risk Probability: {spoof_risk:.2f}%")
print(f"Verdict: {'SPOOF / AI-GENERATED' if spoof_risk >= 50 else 'GENUINE HUMAN'}")
```
