"""
ssl_head.py - Pretrained Self-Supervised Speech model (WavLM / wav2vec2) for SER.

Path C of blueprint:
  Backbone: microsoft/wavlm-base-plus, facebook/wav2vec2-base, or facebook/wav2vec2-xls-r-300m
  Pooling: Temporal mean pooling over transformer last_hidden_state
  Head: Dropout -> Linear (5 classes)
  Freeze feature encoder CNN by default.
"""

from __future__ import annotations

import sys
# Bypass Windows Triton / TorchAO inductor import incompatibility in transformers
sys.modules["torchao"] = None

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class EmotionSSL(nn.Module):
    """
    Self-Supervised Learning (SSL) model for Speech Emotion Recognition.
    """
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        num_classes: int = 5,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        if freeze_encoder and hasattr(self.backbone, "freeze_feature_encoder"):
            self.backbone.freeze_feature_encoder()

        hidden_size = self.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, wav: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        wav: Raw waveform tensor of shape (Batch, Time) float32 at 16 kHz.
        attention_mask: Optional mask of shape (Batch, Time) where 1 is valid, 0 is padding.
        """
        if wav.ndim == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)

        outputs = self.backbone(wav, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (Batch, Frames, Hidden)

        # Mean pooling across temporal frames
        pooled = last_hidden.mean(dim=1)  # (Batch, Hidden)

        logits = self.head(pooled)
        return logits

    def get_parameter_groups(self, lr_backbone: float = 2e-5, lr_head: float = 1e-3, weight_decay: float = 0.01) -> List[Dict]:
        """Separate backbone and classification head parameter groups for differential LR."""
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]
