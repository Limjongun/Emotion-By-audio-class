"""
predict.py - Standalone inference script for Speech Emotion Recognition (SER).

Supports:
  - Single audio file or folder of audio files
  - Sliding window inference for long audio (4s window, 2s hop, averaged softmax probabilities)
  - Loads PyTorch deep learning models (EmotionCNN / EmotionSSL) or Scikit-Learn baseline
  - Outputs predicted emotion class, confidence percentage, and full probability distribution
"""

from __future__ import annotations

import argparse
import json
import sys
sys.modules["torchao"] = None
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Dict, List, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch

from src.data.dataset import EMOTION_CLASSES, LABEL_MAP
from src.features.handcrafted import extract_features_from_waveform
from src.models.cnn import EmotionCNN
from src.models.ssl_head import EmotionSSL


class EmotionPredictor:
    """Wrapper for running inference on arbitrary audio files."""
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        sr: int = 16000,
        window_sec: float = 4.0,
        hop_sec: float = 2.0,
        trim_db: float = 30.0,
    ):
        self.ckpt_path = Path(checkpoint_path).resolve()
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        self.sr = sr
        self.window_samples = int(window_sec * sr)
        self.hop_samples = int(hop_sec * sr)
        self.trim_db = trim_db
        self.classes = EMOTION_CLASSES

        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu")

        # Load model
        if self.ckpt_path.suffix in (".joblib", ".pkl"):
            self.model_type = "handcrafted"
            pkg = joblib.load(self.ckpt_path)
            self.model = pkg["model"]
            self.scaler = pkg["scaler"]
        else:
            self.model_type = "deep_learning"
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            config = ckpt.get("config", {})
            m_cfg = config.get("model", {})
            m_type = m_cfg.get("type", "cnn").lower()

            if m_type == "cnn":
                self.model = EmotionCNN(
                    num_classes=len(self.classes),
                    sample_rate=self.sr,
                    dropout=0.0,
                    use_specaugment=False,
                )
            else:
                self.model = EmotionSSL(
                    model_name=m_cfg.get("name", "microsoft/wavlm-base-plus"),
                    num_classes=len(self.classes),
                    dropout=0.0,
                    freeze_encoder=True,
                )

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

    def preprocess_audio(self, wav_path: str | Path) -> np.ndarray:
        """Load, convert to mono, resample, trim silence, and normalize audio."""
        wav, orig_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0 if wav.shape[0] < wav.shape[1] else 1)

        if orig_sr != self.sr:
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=self.sr)

        # Silence trim
        if self.trim_db and len(wav) > self.sr * 0.2:
            try:
                trimmed, _ = librosa.effects.trim(wav, top_db=self.trim_db)
                if len(trimmed) > self.sr * 0.2:
                    wav = trimmed
            except Exception:
                pass

        # Peak normalization
        peak = np.max(np.abs(wav))
        if peak > 1e-6:
            wav = (wav / peak) * 0.95

        return wav

    def predict(self, wav_path: str | Path) -> Dict[str, Any]:
        """Predict emotion for a given audio file."""
        wav = self.preprocess_audio(wav_path)
        total_len = len(wav)

        # Slice into 4-second windows with 2-second hop (or pad if shorter than 4s)
        windows: List[np.ndarray] = []
        if total_len <= self.window_samples:
            pad_amount = self.window_samples - total_len
            left = pad_amount // 2
            right = pad_amount - left
            padded = np.pad(wav, (left, right), mode="constant", constant_values=0.0)
            windows.append(padded)
        else:
            start = 0
            while start + self.window_samples <= total_len:
                windows.append(wav[start : start + self.window_samples])
                start += self.hop_samples
            if start < total_len and total_len - start > self.hop_samples // 2:
                # Add final window aligned to end
                windows.append(wav[-self.window_samples:])

        # Inference per window
        window_probs: List[np.ndarray] = []

        if self.model_type == "handcrafted":
            for win in windows:
                feat = extract_features_from_waveform(win, sr=self.sr)
                feat_scaled = self.scaler.transform(feat.reshape(1, -1))
                p = self.model.predict_proba(feat_scaled)[0]
                window_probs.append(p)
        else:
            win_tensor = torch.from_numpy(np.stack(windows, axis=0).astype(np.float32)).to(self.device)
            with torch.no_grad():
                logits = self.model(win_tensor)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                window_probs = [p[i] for i in range(len(p))]

        # Average probabilities across windows
        avg_probs = np.mean(window_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))
        pred_emotion = self.classes[pred_idx]
        confidence = float(avg_probs[pred_idx])

        return {
            "path": str(wav_path),
            "predicted_emotion": pred_emotion,
            "confidence": round(confidence * 100.0, 2),
            "probabilities": {cls: round(float(avg_probs[i]) * 100.0, 2) for i, cls in enumerate(self.classes)},
            "n_windows": len(windows),
            "duration_sec": round(total_len / float(self.sr), 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or baseline_model.joblib")
    parser.add_argument("--input", type=str, required=True, help="Path to audio file (.wav) or folder of files")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    args = parser.parse_args()

    predictor = EmotionPredictor(args.ckpt, device=args.device)

    inp_path = Path(args.input)
    if inp_path.is_file():
        files = [inp_path]
    elif inp_path.is_dir():
        files = sorted(inp_path.glob("**/*.wav"))
    else:
        raise FileNotFoundError(f"Input path not found: {inp_path}")

    print(f"\n[*] Running emotion prediction on {len(files)} file(s)...\n")
    results = []
    for f in files:
        res = predictor.predict(f)
        results.append(res)
        print(f"File: {res['path']}")
        print(f"  -> Predicted:  {res['predicted_emotion'].upper()} ({res['confidence']}%)")
        print(f"  -> Probabilities: {res['probabilities']}")
        print(f"  -> Duration: {res['duration_sec']}s ({res['n_windows']} window(s))\n")


if __name__ == "__main__":
    main()
