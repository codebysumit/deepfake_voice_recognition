"""
deepfake_voice_recognition
--------------------------
High-accuracy Deepfake Audio and Voice Cloning Detection Package.
Built on fine-tuned Wav2Vec 2.0 representations paired with a 1D ResNet temporal classification head.

Modules:
    - config: Centralized hyperparameters, audio specs, and dataset paths.
    - segmenter: Audio dataset segmentation, resampling, and preprocessing.
    - dataset: PyTorch dataset loader, balanced splitting, and waveform transforms.
    - model: Wav2Vec2 backbone + 1D ResNet classification head.
    - train: Resumable mixed-precision (AMP) training pipeline.
    - evaluate: Test set evaluation, classification metrics, and confusion matrix.
    - predict: Single-file deepfake risk scoring and probability estimation.
    - export_onnx: FP32 and INT8 quantized ONNX export for low-latency CPU deployment.
    - utils: Checkpoint persistence, training curves, and seeding utilities.
"""

from .config import Config, cfg
from .segmenter import (
    AudioSegmenter,
    find_audio_files,
    process_dataset,
    split_into_chunks,
)

try:
    from .model import (
        ResBlock1D,
        ResNet1DHead,
        Wav2Vec2ResNetDetector,
        build_model,
        count_parameters,
    )
    from .dataset import (
        VoiceDataset,
        build_dataloaders,
        get_test_dataloader,
        scan_dataset,
        visualize_samples,
    )
    from .predict import get_risk_score
except ImportError as _err:
    # Model/Dataset imports require torch/transformers
    Wav2Vec2ResNetDetector = None
    ResNet1DHead = None
    ResBlock1D = None
    build_model = None
    count_parameters = None
    VoiceDataset = None
    scan_dataset = None
    build_dataloaders = None
    get_test_dataloader = None
    visualize_samples = None
    get_risk_score = None

__all__ = [
    "cfg",
    "Config",
    "Wav2Vec2ResNetDetector",
    "ResNet1DHead",
    "ResBlock1D",
    "build_model",
    "count_parameters",
    "VoiceDataset",
    "scan_dataset",
    "build_dataloaders",
    "get_test_dataloader",
    "visualize_samples",
    "AudioSegmenter",
    "find_audio_files",
    "process_dataset",
    "split_into_chunks",
    "get_risk_score",
]

__version__ = "1.0.0"