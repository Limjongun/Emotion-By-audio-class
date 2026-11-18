"""
handcrafted.py - Handcrafted acoustic feature extraction for Baseline A (SVM / Random Forest).

Features extracted per utterance:
  - 40 MFCCs + Deltas + Delta-Deltas (120 acoustic trajectories)
  - Fundamental Frequency (F0 / pitch via pyin/yin)
  - RMS Energy & Zero Crossing Rate (ZCR)
  - Spectral Centroid, Rolloff, and Flatness
  - Summary statistics across time: mean, std, min, max, median, skewness
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew


def compute_summary_stats(series: np.ndarray, prefix: str) -> Dict[str, float]:
    """Calculate mean, std, min, max, median, and skew of a 1D sequence."""
    clean = series[np.isfinite(series)]
    if len(clean) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_skew": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(clean)),
        f"{prefix}_std": float(np.std(clean)),
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_max": float(np.max(clean)),
        f"{prefix}_median": float(np.median(clean)),
        f"{prefix}_skew": float(skew(clean)) if len(clean) > 2 else 0.0,
    }


def extract_features_from_waveform(
    wav: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Extract comprehensive acoustic feature vector from 1D audio waveform.
    Returns: 1D numpy array of concatenated summary statistics.
    """
    features: List[float] = []

    # 1. MFCC + Delta + Delta-Delta
    mfcc = librosa.feature.mfcc(
        y=wav, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    for i in range(n_mfcc):
        features.extend([
            float(np.mean(mfcc[i])),
            float(np.std(mfcc[i])),
            float(np.min(mfcc[i])),
            float(np.max(mfcc[i])),
        ])
        features.extend([
            float(np.mean(delta_mfcc[i])),
            float(np.std(delta_mfcc[i])),
        ])
        features.extend([
            float(np.mean(delta2_mfcc[i])),
            float(np.std(delta2_mfcc[i])),
        ])

    # 2. RMS Energy
    rms = librosa.feature.rms(y=wav, frame_length=win_length, hop_length=hop_length)[0]
    features.extend([
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.min(rms)),
        float(np.max(rms)),
    ])

    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=win_length, hop_length=hop_length)[0]
    features.extend([
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.max(zcr)),
    ])

    # 4. Spectral Centroid, Rolloff, Flatness
    cent = librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    flat = librosa.feature.spectral_flatness(y=wav, n_fft=n_fft, hop_length=hop_length)[0]

    for arr in [cent, rolloff, flat]:
        features.extend([
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
        ])

    # 5. Pitch / F0 (Fundamental frequency)
    try:
        # Fast YIN pitch tracking with hop=640 (40ms)
        f0 = librosa.yin(wav, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=640)
        voiced_f0 = f0[np.isfinite(f0) & (f0 > 50) & (f0 < 500)]
        if len(voiced_f0) > 0:
            features.extend([
                float(np.mean(voiced_f0)),
                float(np.std(voiced_f0)),
                float(np.min(voiced_f0)),
                float(np.max(voiced_f0)),
                float(len(voiced_f0) / len(f0)),  # Voiced ratio
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    feat_arr = np.array(features, dtype=np.float32)
    # Replace any NaN or Inf with 0.0
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_arr


def extract_features_from_df(
    df: pd.DataFrame,
    target_sr: int = 16000,
    crop_samples: int = 64000,
    trim_db: float = 30.0,
    normalize: str = "peak",
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Extract feature matrix X and label vector y from a dataset DataFrame.
    """
    from src.data.dataset import load_and_preprocess_wav

    if max_samples is not None and max_samples < len(df):
        df = df.iloc[:max_samples]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_list: List[Dict[str, Any]] = []

    total = len(df)
    if verbose:
        print(f"[*] Extracting handcrafted acoustic features for {total} files...")

    for idx in range(total):
        row = df.iloc[idx]
        wav_t = load_and_preprocess_wav(
            row["path"],
            target_sr=target_sr,
            crop_samples=crop_samples,
            trim_db=trim_db,
            normalize=normalize,
            is_train=False,
        )
        wav_np = wav_t.numpy()
        feat = extract_features_from_waveform(wav_np, sr=target_sr)

        X_list.append(feat)
        y_list.append(int(row["label_id"]))
        meta_list.append({
            "path": row["path"],
            "speaker": str(row["speaker"]),
            "lang": str(row["lang"]),
            "emotion": str(row["emotion"]),
        })

        if verbose and (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)

    if verbose:
        print(f"[+] Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    return X, y, meta_list
