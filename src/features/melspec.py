"""
melspec.py - Log-Mel Spectrogram extraction module for Path B (CNN).

Converts raw waveform (16 kHz mono) into Log-Mel Spectrogram representations:
  - Sample Rate: 16,000 Hz
  - Window length: 400 (25 ms)
  - Hop length: 160 (10 ms)
  - FFT size: 1024
  - Mel bands: 80 (or 64-80)
  - SpecAugment: Frequency and Time masking
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T

from src.data.augment import SpecAugment


class LogMelSpectrogram(nn.Module):
    """
    Extracts Log-Mel Spectrogram from raw audio waveform.
    Input: (Batch, Time) or (Batch, 1, Time)
    Output: (Batch, 1, n_mels, Time_frames)
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        use_specaugment: bool = False,
    ):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )
        self.use_specaugment = use_specaugment
        self.spec_augment = SpecAugment(
            freq_mask_param=12,
            time_mask_param=24,
            n_freq_masks=2,
            n_time_masks=2,
            p=0.5,
        ) if use_specaugment else None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: Tensor of shape (Batch, Time) or (Batch, 1, Time)
        Returns: Tensor of shape (Batch, 1, n_mels, n_frames)
        """
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)  # (Batch, 1, Time)

        # mel_transform expects (Batch, Channel, Time) or (Batch, Time)
        # Squeeze channel dim for MelSpectrogram then unsqueeze back
        mel = self.mel_transform(wav.squeeze(1))  # (Batch, n_mels, n_frames)

        # Log compression: log(mel + 1e-6)
        log_mel = torch.log(torch.clamp(mel, min=1e-6))

        # Channel normalization: zero mean, unit variance per utterance
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        norm_mel = (log_mel - mean) / std

        # Apply SpecAugment if in training mode
        if self.training and self.spec_augment is not None:
            norm_mel = self.spec_augment(norm_mel)

        return norm_mel.unsqueeze(1)  # (Batch, 1, n_mels, n_frames)
