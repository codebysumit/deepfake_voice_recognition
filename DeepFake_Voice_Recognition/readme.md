# Wav2Vec2 + 1D ResNet — Deepfake Voice Detector

Converted from a single Colab notebook into a proper importable project.
A wav2vec2 backbone (last N transformer layers fine-tuned) feeds a shallow
1D ResNet head that classifies audio as **genuine** or **spoof** (AI voice
clone).

## Project layout

```
deepfake_voice_detector/
├── config.py        # every path/hyperparameter, as a single `cfg` object
├── dataset.py        # VoiceDataset, scan_dataset, build_dataloaders, get_test_dataloader
├── model.py          # ResBlock1D, ResNet1DHead, Wav2Vec2ResNetDetector
├── utils.py          # checkpoint save/load, plot_history, plot_confusion_matrix
├── train.py           # resumable AMP training loop -> run_training()
├── evaluate.py        # classification report + confusion matrix
├── predict.py         # single-file inference -> get_risk_score()
├── export_onnx.py      # ONNX export + INT8 dynamic quantization for CPU
├── main.py            # unified CLI: train / evaluate / predict / export
├── __init__.py        # makes the folder importable as a package
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Expected dataset layout (edit `cfg.dataset_root` in `config.py`, or pass
`--dataset-root`):

```
<dataset_root>/
    real/<lang>/*.wav
    fake/<lang>/*.wav
```

## Usage

Everything can be driven through `main.py`:

```bash
# Train (auto-resumes from checkpoints/checkpoint_latest.pt if it exists)
python main.py train --dataset-root /path/to/data --epochs 40

# Evaluate the best checkpoint on a held-out test split
python main.py evaluate --checkpoint checkpoints/checkpoint_best.pt

# Get a risk score for one audio file
python main.py predict --audio sample.wav --checkpoint checkpoints/checkpoint_best.pt

# Export to ONNX + INT8-quantize for fast CPU inference (e.g. a laptop)
python main.py export --checkpoint checkpoints/checkpoint_best.pt --audio sample.wav
```

Each stage also runs standalone, e.g. `python train.py`, `python predict.py --audio sample.wav`.

## Using it as a module in another project

```python
from deepfake_voice_detector import Wav2Vec2ResNetDetector, cfg
from deepfake_voice_detector.utils import load_model_for_inference
from deepfake_voice_detector.predict import get_risk_score

model, _ = load_model_for_inference(Wav2Vec2ResNetDetector, checkpoint_path="checkpoints/checkpoint_best.pt")
score, label = get_risk_score(model, cfg.device, "sample.wav")
```

## Notes carried over from the original notebook

- `model.preprocess()` is kept **outside** `forward()` on purpose — HF's
  numpy-based feature extractor isn't traceable, so `forward()` stays a
  pure tensor-in/tensor-out function for ONNX/TorchScript export.
- ONNX quantization only targets `MatMul` ops. Quantizing the `Conv1d`
  layers as well made CPU inference noticeably *slower* for this
  architecture — `export_onnx.py` quantizes MatMul-only and warns if
  `QLinearConv` shows up in the exported graph.
- Checkpoints store the full training state (model, optimizer, AMP
  scaler, epoch, best val accuracy, metric history), so training can be
  safely interrupted and resumed by re-running `train.py`.