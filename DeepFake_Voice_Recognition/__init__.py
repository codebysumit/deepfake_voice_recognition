"""
deepfake_voice_detector
------------------------
Wav2Vec2 + 1D ResNet AI-voice / deepfake-audio detector.

Exposes the main building blocks so other code can do:

    from deepfake_voice_detector import Wav2Vec2ResNetDetector, cfg
    from deepfake_voice_detector.predict import get_risk_score
"""

try:
    # When used as an installed package: from deepfake_voice_detector import ...
    from .config import cfg
    from .model import Wav2Vec2ResNetDetector, ResNet1DHead, ResBlock1D, build_model
    from .dataset import VoiceDataset, scan_dataset, build_dataloaders, get_test_dataloader
except ImportError:
    # When scripts are run directly inside the directory: python train.py / python predict.py
    from config import cfg
    from model import Wav2Vec2ResNetDetector, ResNet1DHead, ResBlock1D, build_model
    from dataset import VoiceDataset, scan_dataset, build_dataloaders, get_test_dataloader

__all__ = [
    "cfg",
    "Wav2Vec2ResNetDetector",
    "ResNet1DHead",
    "ResBlock1D",
    "build_model",
    "VoiceDataset",
    "scan_dataset",
    "build_dataloaders",
    "get_test_dataloader",
]

__version__ = "1.0.0"