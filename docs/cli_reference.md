# CLI Reference Guide

The package provides a unified CLI available through `python -m deepfake_voice_recognition.main <command>` or the installed command `deepfake-voice-recognition <command>`.

---

## Commands Summary

| Command | Description |
| :--- | :--- |
| `segment` | Slices raw audio files into uniform chunks and standardizes sample rates. |
| `train` | Runs or resumes mixed-precision (AMP) training with checkpointing. |
| `evaluate` | Evaluates checkpoint on the test set and outputs metrics & confusion matrix. |
| `predict` | Predicts deepfake probability ("risk score") for a single audio file. |
| `export` | Exports PyTorch checkpoint to FP32 and INT8 quantized ONNX models. |

---

## 1. `segment`

```bash
deepfake-voice-recognition segment \
    --source-root ./data/raw \
    --output-root ./data/segmented \
    --segment-seconds 4.0 \
    --sample-rate 16000
```

### Options
- `--source-root`: Path to raw audio folder containing `real/` and `fake/` directories.
- `--output-root`: Target directory to save processed WAV chunks.
- `--segment-seconds`: Chunk duration (default: `4.0`).
- `--sample-rate`: Target sampling rate in Hz (default: `16000`).

---

## 2. `train`

```bash
deepfake-voice-recognition train \
    --dataset-root ./data/segmented \
    --checkpoint-dir ./checkpoints \
    --epochs 40 \
    --batch-size 16 \
    --lr 2e-5
```

### Options
- `--dataset-root`: Root folder of segmented training dataset.
- `--checkpoint-dir`: Directory where checkpoint `.pt` files and logs will be saved.
- `--epochs`: Maximum number of training epochs (default: `40`).
- `--batch-size`: Batch size per step (default: `16`).
- `--lr`: AdamW learning rate (default: `2e-5`).

> **Note on Resuming**: If `checkpoints/checkpoint_latest.pt` exists, the training loop automatically restores model weights, optimizer state, AMP scaler, and metric history to resume seamlessly.

---

## 3. `evaluate`

```bash
deepfake-voice-recognition evaluate \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --dataset-root ./data/segmented \
    --batch-size 16 \
    --test-split 0.20
```

### Options
- `--checkpoint`: Path to `.pt` checkpoint file (default: `./checkpoints/checkpoint_best.pt`).
- `--dataset-root`: Root folder of dataset.
- `--test-split`: Held-out test proportion (default: `0.20`).

---

## 4. `predict`

```bash
deepfake-voice-recognition predict \
    --audio ./samples/sample_test.wav \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --verbose
```

### Options
- `--audio`: Path to input `.wav`, `.mp3`, or `.flac` audio file.
- `--checkpoint`: Trained checkpoint path.
- `--verbose`: Prints raw model logits and softmax probabilities.

---

## 5. `export`

```bash
deepfake-voice-recognition export \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --audio ./samples/sample_test.wav
```

### Options
- `--checkpoint`: Checkpoint to export.
- `--audio`: (Optional) Sample audio to trace shape and verify execution.
- `--skip-quantize`: Skips dynamic INT8 quantization and outputs only FP32 ONNX.
