"""
train.py - Unified Training Pipeline for Speech Emotion Recognition (SER).

Supports 3 model pathways:
  1. handcrafted: Baseline A (MFCC + F0 + Energy stats -> StandardScaler -> SVM/RF)
  2. cnn: Path B (Log-Mel Spectrogram -> EmotionCNN / ResNet)
  3. wavlm / ssl: Path C (Pretrained WavLM / wav2vec2 -> EmotionSSL)

Features:
  - Anti-leakage verification at startup
  - Automatic Mixed Precision (AMP fp16)
  - Gradient Accumulation
  - Cosine / OneCycle scheduler
  - Early stopping strictly on Validation UAR (Balanced Accuracy)
  - TensorBoard logging
  - Automatic evaluation on Test split and generation of comprehensive report
"""

from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
# Bypass Windows Triton / TorchAO bug
sys.modules["torchao"] = None

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import EMOTION_CLASSES, LABEL_MAP, EmotionDataset
from src.data.make_split import verify_split_files
from src.features.handcrafted import extract_features_from_df
from src.metrics import compute_metrics, save_report
from src.models.cnn import EmotionCNN
from src.models.ssl_head import EmotionSSL


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_handcrafted_baseline(
    config: Dict[str, Any],
    run_dir: Path,
) -> None:
    """Train Baseline A: Handcrafted features + SVM / Random Forest."""
    print("\n" + "=" * 60)
    print("TRAINING JALUR A: HANDCRAFTED BASELINE (SVM / RF)")
    print("=" * 60)

    train_csv = config["data"]["train_csv"]
    val_csv = config["data"]["val_csv"]
    test_csv = config["data"]["test_csv"]

    train_df = pd.read_csv(train_csv, dtype={"speaker": str, "utt_id": str})
    val_df = pd.read_csv(val_csv, dtype={"speaker": str, "utt_id": str})
    test_df = pd.read_csv(test_csv, dtype={"speaker": str, "utt_id": str})

    sr = config["data"].get("sr", 16000)
    crop_samples = int(config["data"].get("crop_sec", 4.0) * sr)
    trim_db = config["data"].get("trim_db", 30.0)

    # Anti-leakage rule: Extract features and fit scaler ONLY on train!
    print("[*] Extracting Train features...")
    X_train, y_train, meta_train = extract_features_from_df(train_df, target_sr=sr, crop_samples=crop_samples, trim_db=trim_db)
    print("[*] Extracting Val features...")
    X_val, y_val, meta_val = extract_features_from_df(val_df, target_sr=sr, crop_samples=crop_samples, trim_db=trim_db)
    print("[*] Extracting Test features...")
    X_test, y_test, meta_test = extract_features_from_df(test_df, target_sr=sr, crop_samples=crop_samples, trim_db=trim_db)

    print("[*] Fitting StandardScaler ONLY on Train split...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM classifier
    clf_type = config.get("model", {}).get("classifier", "svm").lower()
    if clf_type == "rf":
        print("[*] Training Random Forest Classifier...")
        clf = RandomForestClassifier(n_estimators=300, random_state=config.get("seed", 42), n_jobs=-1)
    else:
        print("[*] Training RBF SVM Classifier (with probability estimates)...")
        clf = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=config.get("seed", 42))

    t0 = time.time()
    clf.fit(X_train_scaled, y_train)
    print(f"[+] Fitting completed in {time.time() - t0:.2f} seconds.")

    # Save model and scaler
    joblib.dump({"model": clf, "scaler": scaler, "config": config}, run_dir / "baseline_model.joblib")

    # Evaluate on Val
    val_probs = clf.predict_proba(X_val_scaled)
    val_preds = np.argmax(val_probs, axis=1)
    val_metrics = compute_metrics(
        y_val, val_preds, EMOTION_CLASSES, y_prob=val_probs,
        speakers=[m["speaker"] for m in meta_val],
        langs=[m["lang"] for m in meta_val],
    )
    print(f"\n[+] Validation UAR: {val_metrics['uar']*100:.2f}% | Macro-F1: {val_metrics['macro_f1']*100:.2f}%")
    save_report(
        val_metrics, run_dir / "report_val", y_true=y_val, y_pred=val_preds,
        class_names=EMOTION_CLASSES, y_prob=val_probs,
        paths=[m["path"] for m in meta_val],
        speakers=[m["speaker"] for m in meta_val],
        langs=[m["lang"] for m in meta_val],
    )

    # Evaluate on Test
    test_probs = clf.predict_proba(X_test_scaled)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = compute_metrics(
        y_test, test_preds, EMOTION_CLASSES, y_prob=test_probs,
        speakers=[m["speaker"] for m in meta_test],
        langs=[m["lang"] for m in meta_test],
    )
    print(f"[+] Test UAR:       {test_metrics['uar']*100:.2f}% | Macro-F1: {test_metrics['macro_f1']*100:.2f}% | MCC: {test_metrics['mcc']:.4f}")
    save_report(
        test_metrics, run_dir / "report_test", y_true=y_test, y_pred=test_preds,
        class_names=EMOTION_CLASSES, y_prob=test_probs,
        paths=[m["path"] for m in meta_test],
        speakers=[m["speaker"] for m in meta_test],
        langs=[m["lang"] for m in meta_test],
    )

    print(f"[+] Baseline A run complete. Artifacts saved in: {run_dir}")


def train_deep_learning(
    config: Dict[str, Any],
    run_dir: Path,
) -> None:
    """Train Deep Learning models: CNN (Path B) or SSL/WavLM (Path C)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config["model"]

    # 1. Datasets & Loaders
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 0)  # Safe for Windows

    train_ds = EmotionDataset(
        data_cfg["train_csv"],
        target_sr=data_cfg.get("sr", 16000),
        crop_sec=data_cfg.get("crop_sec", 4.0),
        trim_db=data_cfg.get("trim_db", 30.0),
        split_type="train",
        use_augmentation=train_cfg.get("use_augmentation", True),
    )
    val_ds = EmotionDataset(
        data_cfg["val_csv"],
        target_sr=data_cfg.get("sr", 16000),
        crop_sec=data_cfg.get("crop_sec", 4.0),
        trim_db=data_cfg.get("trim_db", 30.0),
        split_type="val",
        use_augmentation=False,
    )
    test_ds = EmotionDataset(
        data_cfg["test_csv"],
        target_sr=data_cfg.get("sr", 16000),
        crop_sec=data_cfg.get("crop_sec", 4.0),
        trim_db=data_cfg.get("trim_db", 30.0),
        split_type="test",
        use_augmentation=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # 2. Model Creation
    model_type = model_cfg.get("type", "cnn").lower()
    num_classes = len(data_cfg.get("classes", EMOTION_CLASSES))

    if model_type == "cnn":
        print("[*] Instantiating EmotionCNN (Log-Mel + ResNet blocks)...")
        model = EmotionCNN(
            num_classes=num_classes,
            n_mels=model_cfg.get("n_mels", 80),
            sample_rate=data_cfg.get("sr", 16000),
            dropout=model_cfg.get("dropout", 0.3),
            use_specaugment=train_cfg.get("use_specaugment", True),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
    elif model_type in ("wavlm", "ssl", "wav2vec2"):
        model_name = model_cfg.get("name", "microsoft/wavlm-base-plus")
        print(f"[*] Instantiating EmotionSSL with backbone: {model_name}...")
        model = EmotionSSL(
            model_name=model_name,
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.2),
            freeze_encoder=model_cfg.get("freeze_encoder", True),
        )
        param_groups = model.get_parameter_groups(
            lr_backbone=float(train_cfg.get("lr_backbone", 2e-5)),
            lr_head=float(train_cfg.get("lr_head", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    model = model.to(device)

    # 3. Criterion, AMP & Scheduler
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    use_amp = bool(train_cfg.get("amp", True)) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum = int(train_cfg.get("grad_accum", 1))

    epochs = int(train_cfg.get("epochs", 10))
    patience = int(train_cfg.get("patience", 6))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    except Exception:
        writer = None

    # 4. Training Loop
    best_val_uar = 0.0
    patience_counter = 0
    best_ckpt_path = run_dir / "best.pt"

    print("\n" + "=" * 60)
    print(f"STARTING TRAINING: {epochs} Epochs | Batch Size {batch_size} (Accum {grad_accum}) | AMP: {use_amp}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            wavs = batch["wav"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(wavs)
                loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        scheduler.step()

        # Validation phase
        val_loss, val_uar, val_acc, val_f1, _, _, _ = evaluate_model(model, val_loader, criterion, device, use_amp)

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("UAR/val", val_uar, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("Macro_F1/val", val_f1, epoch)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val UAR: {val_uar*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val F1: {val_f1*100:.2f}%"
        )

        # Early stopping and checkpointing strictly on Validation UAR
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_uar": val_uar,
                "config": config,
            }, best_ckpt_path)
            print(f"  --> Checkpoint saved: new best Val UAR = {val_uar*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Early stopping triggered after {patience} epochs without val UAR improvement.")
                break

    if writer is not None:
        writer.close()

    # 5. Final Evaluation on Best Checkpoint
    print("\n" + "=" * 60)
    print("FINAL EVALUATION WITH BEST CHECKPOINT (SPEAKER-INDEPENDENT TEST)")
    print("=" * 60)

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[+] Loaded best checkpoint from epoch {ckpt['epoch']} (Val UAR: {ckpt['val_uar']*100:.2f}%)")

    # Evaluate on Test set
    _, test_uar, test_acc, test_f1, y_true_test, y_pred_test, y_prob_test = evaluate_model(
        model, test_loader, criterion, device, use_amp
    )

    test_meta = [test_ds[i] for i in range(len(test_ds))]
    test_speakers = [m["speaker"] for m in test_meta]
    test_langs = [m["lang"] for m in test_meta]
    test_paths = [m["path"] for m in test_meta]

    test_metrics = compute_metrics(
        y_true_test,
        y_pred_test,
        class_names=data_cfg.get("classes", EMOTION_CLASSES),
        y_prob=y_prob_test,
        speakers=test_speakers,
        langs=test_langs,
    )

    report_text = save_report(
        test_metrics,
        out_dir=run_dir / "report_test",
        y_true=y_true_test,
        y_pred=y_pred_test,
        class_names=data_cfg.get("classes", EMOTION_CLASSES),
        y_prob=y_prob_test,
        paths=test_paths,
        speakers=test_speakers,
        langs=test_langs,
    )

    print(report_text)
    print(f"\n[+] Full test report saved to: {run_dir / 'report_test'}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate PyTorch model over a DataLoader.
    Returns: (avg_loss, uar, accuracy, macro_f1, y_true, y_pred, y_prob)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in loader:
        wavs = batch["wav"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(wavs)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        total_loss += loss.item()
        n_batches += 1

        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / max(1, n_batches)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    uar = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(np.mean(y_true == y_pred))
    from sklearn.metrics import f1_score
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return avg_loss, uar, acc, f1, y_true, y_pred, y_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition Pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--run-name", type=str, default="", help="Experiment run name (folder inside runs/)")
    parser.add_argument("--model-type", type=str, default="", help="Override model.type (handcrafted | cnn | wavlm)")
    parser.add_argument("--batch-size", type=int, default=0, help="Override train.batch_size")
    parser.add_argument("--epochs", type=int, default=0, help="Override train.epochs")
    parser.add_argument("--no-verify", action="store_true", help="Skip anti-leakage verification at start")

    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model_type:
        config["model"]["type"] = args.model_type
    if args.batch_size > 0:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs > 0:
        config["train"]["epochs"] = args.epochs

    seed = config.get("seed", 42)
    set_seed(seed)

    # 1. Anti-leakage verification
    if not args.no_verify:
        print("[*] Running pre-training Anti-Leakage verification...")
        verify_split_files(
            config["data"]["train_csv"],
            config["data"]["val_csv"],
            config["data"]["test_csv"],
            verbose=True,
        )

    # 2. Setup run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"].lower()
    run_name = args.run_name if args.run_name else f"{model_type}_{timestamp}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy config and label_map.json to run_dir
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, indent=2)

    with open(run_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    # Copy split_report.json if exists
    split_rep = Path(config["data"]["train_csv"]).parent / "split_report.json"
    if split_rep.exists():
        shutil.copy(split_rep, run_dir / "split_report.json")

    print(f"[+] Initialized experiment run directory: {run_dir}")

    # 3. Route to training pathway
    if model_type == "handcrafted":
        train_handcrafted_baseline(config, run_dir)
    else:
        train_deep_learning(config, run_dir)


if __name__ == "__main__":
    main()
