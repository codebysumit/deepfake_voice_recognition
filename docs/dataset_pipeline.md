# Dataset Pipeline & Preprocessing

This guide covers dataset preparation, automatic audio segmentation, resampling, and DataLoader batching for **deepfake_voice_recognition**.

---

## 1. Directory Structure Conventions

### Raw Input Directory Structure
Organize your raw audio files under `real` (or `genuine`) and `fake` (or `spoof`, `clone`) directories, optionally subdivided by language or speaker:

```
data/raw/
├── fake/
│   ├── Bengali/
│   │   ├── 101_Gemini_TTS.mp3
│   │   └── 102_ElevenLabs.wav
│   ├── English/
│   │   └── 103_VALL-E.flac
│   └── Hindi/
│       └── 104_Coqui_TTS.wav
└── real/
    ├── Bengali/
    │   ├── 001_speakerA.wav
    │   └── 002_speakerB.mp3
    ├── English/
    │   └── 003_speakerC.wav
    └── Hindi/
        └── 004_speakerD.wav
```

---

## 2. Audio Segmentation (`AudioSegmenter`)

Heterogeneous audio datasets often feature clips of varying duration. For efficient neural network training:
1. Long clips are segmented into uniform windows of duration `SEGMENT_SECONDS` (default: `4.0` seconds = `64,000` samples at `16 kHz`).
2. Trailing segments shorter than 4 seconds are zero-padded with silence.
3. Every file is converted to mono and resampled to `16,000 Hz`.
4. The output retains the `real/fake` and language hierarchy.

### Running via CLI:
```bash
python -m deepfake_voice_recognition.main segment \
    --source-root ./data/raw \
    --output-root ./data/segmented \
    --segment-seconds 4.0 \
    --sample-rate 16000
```

### Running in Python:
```python
from deepfake_voice_recognition.segmenter import AudioSegmenter

segmenter = AudioSegmenter(segment_seconds=4.0, target_sr=16000)
summary = segmenter.process_dataset(
    source_root="./data/raw",
    output_root="./data/segmented"
)
print(f"Total chunks created: {summary['total_chunks']}")
```

---

## 3. PyTorch `VoiceDataset` & DataLoaders

The dataset loader (`deepfake_voice_recognition.dataset`) provides:
- **`scan_dataset`**: Scans the segmented dataset directory, extracts genuine (0) and spoof (1) labels, and shuffles indices.
- **`build_dataloaders`**: Creates stratified `train_loader` and `val_loader` with CUDA pin-memory and multi-process loading.
- **`get_test_dataloader`**: Constructs a separate, stratified test loader for rigorous out-of-sample evaluation.
