"""
model.py
--------
The network: a wav2vec2 backbone (partially fine-tuned) feeding a shallow
1D ResNet classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from config import cfg


class ResBlock1D(nn.Module):
    """conv -> BN -> ReLU -> conv -> BN, with a skip connection."""

    def __init__(self, in_ch, out_ch, stride=1):
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

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class ResNet1DHead(nn.Module):
    """Shallow ResNet-style stack over the time axis of wav2vec2 embeddings."""

    def __init__(self, in_dim, base_channels=128, num_blocks=(1, 1, 1)):
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
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (batch, time, channels) -> (batch, channels, time)
        x = self.input_proj(x)
        x = self.stages(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.classifier(x)


class Wav2Vec2ResNetDetector(nn.Module):
    """
    wav2vec2 backbone (last `num_unfrozen_layers` transformer layers
    trainable, everything else frozen) + ResNet1DHead classifier.
    """

    def __init__(self, model_name=None, num_unfrozen_layers=None):
        super().__init__()
        model_name = model_name or cfg.model_name
        num_unfrozen_layers = cfg.num_unfrozen_layers if num_unfrozen_layers is None else num_unfrozen_layers

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self._freeze_backbone_layers(num_unfrozen_layers)
        hidden_dim = self.wav2vec2.config.hidden_size
        self.head = ResNet1DHead(in_dim=hidden_dim)

    def _freeze_backbone_layers(self, num_unfrozen_layers):
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        total_layers = len(self.wav2vec2.encoder.layers)
        freeze_until = max(0, total_layers - num_unfrozen_layers)

        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            for param in layer.parameters():
                param.requires_grad = i >= freeze_until

        print(f"wav2vec2: {total_layers} transformer layers total, "
              f"fine-tuning last {num_unfrozen_layers}, rest frozen.")

    def preprocess(self, raw_waveforms, sampling_rate=None):
        """
        Convert a list/batch of raw waveforms into normalized, padded
        input_values. Kept OUTSIDE forward() so forward() stays a pure
        tensor-in/tensor-out function (needed for ONNX/TorchScript export;
        HF's numpy-based feature extractor isn't traceable).
        """
        sampling_rate = sampling_rate or cfg.sample_rate
        inputs = self.feature_extractor(
            [w.cpu().numpy() if torch.is_tensor(w) else w for w in raw_waveforms],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return inputs.input_values

    def forward(self, input_values):
        """
        Args:
            input_values: FloatTensor (batch, time) — already normalized
                           and padded via self.preprocess().
        """
        wav2vec2_out = self.wav2vec2(input_values).last_hidden_state
        return self.head(wav2vec2_out)


def build_model(device=None):
    """Convenience factory: builds the detector and moves it to `device`."""
    device = device or cfg.device
    model = Wav2Vec2ResNetDetector().to(device)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick architecture sanity check: `python model.py`
    m = build_model()
    total, trainable = count_parameters(m)
    print(m)
    print(f"\nTotal Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")