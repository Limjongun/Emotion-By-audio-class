"""
augment.py - Audio and Spectrogram Augmentation for Speech Emotion Recognition.

Per blueprint guidelines:
  - Allowed: SpecAugment (time/frequency masking), subtle additive Gaussian/pink noise, random gain.
  - Strictly restricted/avoided: Pitch shifting and heavy time stretching (distorts emotional prosody).
  - Applied ONLY to training splits.
"""

from __future__ import annotations

import math
import random
import torch
import torch.nn as nn


class RandomGain(nn.Module):
    """Apply random gain in decibels to waveform."""
    def __init__(self, min_gain_db: float = -6.0, max_gain_db: float = 3.0):
        super().__init__()
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        return torch.clamp(wav * gain_linear, -1.0, 1.0)


class AdditiveGaussianNoise(nn.Module):
    """Add low-amplitude Gaussian noise (SNR ~ 15-30 dB)."""
    def __init__(self, snr_min: float = 15.0, snr_max: float = 30.0, p: float = 0.5):
        super().__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.p = p

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return wav

        signal_power = torch.mean(wav ** 2)
        if signal_power < 1e-7:
            return wav

        snr = random.uniform(self.snr_min, self.max_gain_db if hasattr(self, 'max_gain_db') else self.snr_max)
        noise_power = signal_power / (10.0 ** (snr / 10.0))
        noise = torch.randn_like(wav) * torch.sqrt(noise_power)
        return torch.clamp(wav + noise, -1.0, 1.0)


class SpecAugment(nn.Module):
    """
    SpecAugment (Frequency and Time Masking) for spectrograms.
    Input shape: (..., Freq, Time) or (Batch, Channel, Freq, Time)
    """
    def __init__(
        self,
        freq_mask_param: int = 12,
        time_mask_param: int = 24,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: Tensor of shape (..., F, T)
        """
        if not self.training or random.random() > self.p:
            return spec

        cloned = spec.clone()
        *_, n_freq, n_time = cloned.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f_width = random.randint(0, min(self.freq_mask_param, n_freq))
            f_start = random.randint(0, max(0, n_freq - f_width))
            cloned[..., f_start : f_start + f_width, :] = 0.0

        # Time masking
        for _ in range(self.n_time_masks):
            t_width = random.randint(0, min(self.time_mask_param, n_time))
            t_start = random.randint(0, max(0, n_time - t_width))
            cloned[..., :, t_start : t_start + t_width] = 0.0

        return cloned
