"""
model.py
--------
Deepfake Voice Detection Architecture:
Fine-tuned Wav2Vec2 audio representation backbone coupled to a 1D ResNet
temporal feature aggregation and classification head.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

try:
    from .config import cfg
except ImportError:
    from config import cfg


class ResBlock1D(nn.Module):
    """1D Residual Block: Conv1d -> BatchNorm1d -> ReLU -> Conv1d -> BatchNorm1d + Skip Connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class ResNet1DHead(nn.Module):
    """
    1D ResNet Temporal Convolutional Head for processing Wav2Vec2 frame-level embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        base_channels: int = 128,
        num_blocks: Sequence[int] = (1, 1, 1),
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_dim, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        channels = base_channels
        stages = []
        for i, n_blocks in enumerate(num_blocks):
            out_channels = channels * 2 if i > 0 else channels
            stride = 2 if i > 0 else 1
            stages.append(ResBlock1D(channels, out_channels, stride=stride))
            for _ in range(n_blocks - 1):
                stages.append(ResBlock1D(out_channels, out_channels, stride=1))
            channels = out_channels
        self.stages = nn.Sequential(*stages)

        self.classifier = nn.Sequential(
            nn.Linear(channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.stages(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.classifier(x)


class Wav2Vec2ResNetDetector(nn.Module):
    """
    Complete Deepfake Voice Detector.
    Wav2Vec 2.0 backbone (fine-tuning last `num_unfrozen_layers` layers) + ResNet1D classification head.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_unfrozen_layers: Optional[int] = None,
    ):
        super().__init__()
        model_name = model_name or cfg.model_name
        num_unfrozen_layers = cfg.num_unfrozen_layers if num_unfrozen_layers is None else num_unfrozen_layers

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self._freeze_backbone_layers(num_unfrozen_layers)

        hidden_dim = self.wav2vec2.config.hidden_size
        self.head = ResNet1DHead(in_dim=hidden_dim)

    def _freeze_backbone_layers(self, num_unfrozen_layers: int) -> None:
        """Freezes feature encoder and early transformer layers, keeping last N layers trainable."""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        total_layers = len(self.wav2vec2.encoder.layers)
        freeze_until = max(0, total_layers - num_unfrozen_layers)

        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            for param in layer.parameters():
                param.requires_grad = (i >= freeze_until)

        print(
            f"Wav2Vec2 backbone: {total_layers} transformer encoder layers. "
            f"Fine-tuning top {num_unfrozen_layers} layers (bottom {freeze_until} frozen)."
        )

    def preprocess(self, raw_waveforms: Sequence[torch.Tensor], sampling_rate: Optional[int] = None) -> torch.Tensor:
        """
        Normalizes and pads raw 1D waveforms into a batched Tensor.
        Separated from forward() to ensure forward() is cleanly traceable for ONNX export.
        """
        sampling_rate = sampling_rate or cfg.sample_rate
        inputs = self.feature_extractor(
            [w.cpu().numpy() if torch.is_tensor(w) else w for w in raw_waveforms],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return inputs.input_values

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_values: Preprocessed FloatTensor of shape (batch_size, num_samples).
        Returns:
            Logits of shape (batch_size, 2).
        """
        wav2vec2_out = self.wav2vec2(input_values).last_hidden_state
        return self.head(wav2vec2_out)


def build_model(device: Optional[torch.device] = None) -> Wav2Vec2ResNetDetector:
    """Builds and initializes the detector on the target compute device."""
    device = device or cfg.device
    model = Wav2Vec2ResNetDetector().to(device)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    m = build_model()
    total, trainable = count_parameters(m)
    print(m)
    print(f"\nTotal Parameters    : {total:,}")
    print(f"Trainable Parameters: {trainable:,}")