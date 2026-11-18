"""
cnn.py - Lightweight CNN / ResNet architecture for Log-Mel Spectrogram Speech Emotion Recognition.

Path B of blueprint:
  Input: Raw waveform (Batch, Time) OR Log-Mel Spectrogram (Batch, 1, n_mels, frames)
  Feature extraction: Conv2D blocks with Residual connections and BatchNorm
  Classifier: Global Average Pooling -> Dropout -> Linear (5 classes)
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.melspec import LogMelSpectrogram


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> ReLU -> MaxPool"""
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """2x Conv2d + BatchNorm with skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class EmotionCNN(nn.Module):
    """
    CNN / ResNet for 5-class Speech Emotion Recognition on Log-Mel Spectrograms.
    """
    def __init__(
        self,
        num_classes: int = 5,
        n_mels: int = 80,
        sample_rate: int = 16000,
        dropout: float = 0.3,
        use_specaugment: bool = True,
    ):
        super().__init__()
        self.frontend = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            use_specaugment=use_specaugment,
        )

        # Backbone layers: 1 -> 32 -> 64 -> 128 -> 256
        self.layer1 = ConvBlock(1, 32, pool=True)
        self.res1 = ResidualBlock(32)

        self.layer2 = ConvBlock(32, 64, pool=True)
        self.res2 = ResidualBlock(64)

        self.layer3 = ConvBlock(64, 128, pool=True)
        self.res3 = ResidualBlock(128)

        self.layer4 = ConvBlock(128, 256, pool=True)
        self.res4 = ResidualBlock(256)

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Raw waveform of shape (Batch, Time) OR Spectrogram (Batch, 1, Freq, Time)
        """
        if x.ndim == 2 or (x.ndim == 3 and x.shape[1] == 1):
            # Compute log-mel spectrogram
            x = self.frontend(x)

        x = self.layer1(x)
        x = self.res1(x)

        x = self.layer2(x)
        x = self.res2(x)

        x = self.layer3(x)
        x = self.res3(x)

        x = self.layer4(x)
        x = self.res4(x)

        x = self.global_pool(x).flatten(1)
        logits = self.head(x)
        return logits
