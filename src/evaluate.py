"""
evaluate.py - Standalone evaluation script for trained SER checkpoints.

Takes a checkpoint (best.pt or baseline_model.joblib) and a split CSV (e.g. data/test.csv),
runs inference, exports predictions.csv, and compiles comprehensive reports via metrics.py.
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
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import EMOTION_CLASSES, LABEL_MAP, EmotionDataset
from src.features.handcrafted import extract_features_from_df
from src.metrics import compute_metrics, save_report
from src.models.cnn import EmotionCNN
from src.models.ssl_head import EmotionSSL


def evaluate_checkpoint(
    ckpt_path: str | Path,
    csv_path: str | Path,
    out_dir: str | Path,
    device_str: str = "auto",
    batch_size: int = 16,
) -> None:
    ckpt_file = Path(ckpt_path).resolve()
    csv_file = Path(csv_path).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_file}")

    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")

    df = pd.read_csv(csv_file, dtype={"speaker": str, "utt_id": str})
    print(f"[*] Evaluating on {len(df)} samples from: {csv_file}")

    # Check if checkpoint is joblib (Baseline A) or PyTorch (.pt)
    if ckpt_file.suffix in (".joblib", ".pkl"):
        print("[*] Detected Scikit-Learn baseline checkpoint (.joblib)...")
        pkg = joblib.load(ckpt_file)
        model = pkg["model"]
        scaler = pkg["scaler"]
        config = pkg.get("config", {})

        sr = config.get("data", {}).get("sr", 16000)
        crop_samples = int(config.get("data", {}).get("crop_sec", 4.0) * sr)
        trim_db = config.get("data", {}).get("trim_db", 30.0)

        X, y_true, meta = extract_features_from_df(df, target_sr=sr, crop_samples=crop_samples, trim_db=trim_db)
        X_scaled = scaler.transform(X)

        y_prob = model.predict_proba(X_scaled)
        y_pred = np.argmax(y_prob, axis=1)

        speakers = [m["speaker"] for m in meta]
        langs = [m["lang"] for m in meta]
        paths = [m["path"] for m in meta]

    else:
        print(f"[*] Detected PyTorch deep learning checkpoint ({device})...")
        ckpt = torch.load(ckpt_file, map_location=device)
        config = ckpt.get("config", {})
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        model_type = model_cfg.get("type", "cnn").lower()
        num_classes = len(data_cfg.get("classes", EMOTION_CLASSES))

        if model_type == "cnn":
            model = EmotionCNN(
                num_classes=num_classes,
                n_mels=model_cfg.get("n_mels", 80),
                sample_rate=data_cfg.get("sr", 16000),
                dropout=0.0,
                use_specaugment=False,
            )
        else:
            model = EmotionSSL(
                model_name=model_cfg.get("name", "microsoft/wavlm-base-plus"),
                num_classes=num_classes,
                dropout=0.0,
                freeze_encoder=True,
            )

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        dataset = EmotionDataset(
            csv_file,
            target_sr=data_cfg.get("sr", 16000),
            crop_sec=data_cfg.get("crop_sec", 4.0),
            trim_db=data_cfg.get("trim_db", 30.0),
            split_type="test",
            use_augmentation=False,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        all_preds = []
        all_targets = []
        speakers = []
        langs = []
        paths = []

        with torch.no_grad():
            for batch in loader:
                wavs = batch["wav"].to(device)
                logits = model(wavs)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(batch["label"].numpy())
                speakers.extend(batch["speaker"])
                langs.extend(batch["lang"])
                paths.extend(batch["path"])

        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)

    # Compute comprehensive metrics
    classes = config.get("data", {}).get("classes", EMOTION_CLASSES)
    metrics = compute_metrics(
        y_true,
        y_pred,
        class_names=classes,
        y_prob=y_prob,
        speakers=speakers,
        langs=langs,
    )

    # Save reports and annotated predictions CSV
    report_text = save_report(
        metrics,
        out_dir=out_path,
        y_true=y_true,
        y_pred=y_pred,
        class_names=classes,
        y_prob=y_prob,
        paths=paths,
        speakers=speakers,
        langs=langs,
    )

    print(report_text)
    print(f"\n[+] Full evaluation artifacts saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SER checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or baseline_model.joblib")
    parser.add_argument("--csv", type=str, default="data/test.csv", help="Path to evaluation CSV")
    parser.add_argument("--out", type=str, default="reports/evaluation", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    evaluate_checkpoint(
        args.ckpt,
        args.csv,
        args.out,
        device_str=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
