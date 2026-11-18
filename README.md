# Speech Emotion Recognition (SER) Pipeline

End-to-end speaker-independent Speech Emotion Recognition (SER) pipeline built for the **Emotion Speech Dataset (ESD)**, adhering strictly to the design blueprint (`blueprint_ser_pipeline.md`).

---

## 1. Features & Architecture

- **Classes**: 5 emotions (`neutral`, `happy`, `angry`, `sad`, `surprise`).
- **Input**: 16 kHz mono `.wav`, silence-trimmed (`top_db=30`), peak normalized, 4-second fixed window (random crop for training, center crop for evaluation).
- **Speaker-Independent Splitting**: Strict Leave-Speakers-Out splits generated via `src/data/make_split.py` (`fixed`, `stratified`, or `kfold` Group K-Fold).
- **Automated Anti-Leakage Verification**:
  - Verification of no speaker overlap across splits (`train`, `val`, `test`).
  - Detection of path and MD5 hash duplicates.
  - Verification that all 5 classes are present in every split.
  - Language representation checks (`zh` vs `en`).
  - Procedural anti-leakage guarantees (scalers fit only on train, augmentations only on train, tuning only on val).
- **Three Modeling Pathways**:
  - **Path A (Baseline)**: Handcrafted acoustic features (40 MFCCs + deltas + delta-deltas, F0, RMS energy, spectral features, summary statistics) + StandardScaler + SVM (RBF) / Random Forest.
  - **Path B (CNN)**: Log-Mel Spectrogram (80 mels) + SpecAugment + ResNet Conv2D blocks + Global Average Pooling + Linear head.
  - **Path C (SSL Pretrained)**: Pretrained WavLM (`microsoft/wavlm-base-plus`) / wav2vec2 with frozen CNN encoder and temporal mean pooling head.
- **Evaluation Module (`src/metrics.py`)**:
  - Primary metric: **UAR (Unweighted Average Recall / Balanced Accuracy)**.
  - Secondary metrics: WAR (Accuracy), Macro-F1, Weighted-F1, MCC, Cohen's Kappa, ECE (Expected Calibration Error), Log Loss, Top-2 Accuracy, Multiclass ROC-AUC (OvR macro).
  - Confusion Matrix (counts & row-normalized percentage heatmap PNG).
  - Breakdown per speaker (`per_speaker.csv`) and per language (`per_lang.csv`).
  - **Subject-level 95% Bootstrap Confidence Intervals** (resampling speakers with replacement).
  - Cross-corpus evaluation helper with label restriction (`restrict_labels`).
- **Inference (`src/predict.py`)**:
  - Sliding-window inference (4s window, 2s hop, averaged softmax probabilities) for audio files of any length.

---

## 2. Directory Structure

```
ser-pipeline/
├── configs/
│   └── base.yaml                 # Configuration parameters
├── data/
│   ├── manifest.csv              # Full dataset manifest
│   ├── split.csv                 # Manifest with split assignments
│   ├── train.csv, val.csv, test.csv
│   └── split_report.json         # Distribution and anti-leakage report
├── src/
│   ├── data/
│   │   ├── make_split.py         # Split generator + anti-leakage checker
│   │   ├── dataset.py            # PyTorch Dataset + preprocessing
│   │   └── augment.py            # SpecAugment & waveform augmentations
│   ├── features/
│   │   ├── handcrafted.py        # 40 MFCC+F0+energy acoustic feature extraction
│   │   └── melspec.py            # Log-Mel Spectrogram extraction layer
│   ├── models/
│   │   ├── cnn.py                # ResNet/CNN on Log-Mel spectrogram
│   │   └── ssl_head.py           # WavLM/wav2vec2 SSL model with pooling head
│   ├── metrics.py                # Comprehensive metrics & reporting CLI
│   ├── train.py                  # Unified training loop for all 3 pathways
│   ├── evaluate.py               # Standalone checkpoint evaluator
│   └── predict.py                # Production inference (single file / folder)
├── tests/
│   └── test_split_metrics.py     # 16 unit tests for splits, leakage, and metrics
├── runs/                         # Experiment runs (checkpoints, logs, reports)
├── reports/                      # Evaluation reports
├── requirements.txt
└── README.md
```

---

## 3. Quickstart Guide

### 3.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Generate Speaker-Independent Splits
Generate stratified splits (balanced per language: 14 train / 2 val / 4 test):
```bash
python src/data/make_split.py --root "D:/datasets/archive/Emotion Speech Dataset" --out data --mode stratified --seed 42
```
Or generate Group K-Fold splits (5-fold cross-validation):
```bash
python src/data/make_split.py --root "D:/datasets/archive/Emotion Speech Dataset" --out data --mode kfold --k 5 --seed 42
```
Verify an existing split against leakage:
```bash
python src/data/make_split.py --verify data
```

### 3.3 Run Unit Tests
```bash
python -m pytest tests/test_split_metrics.py -v
```

### 3.4 Training Models

**Train Baseline A (Handcrafted + SVM):**
```bash
python src/train.py --model-type handcrafted --run-name baseline_svm
```

**Train Path B (Log-Mel + CNN / ResNet):**
```bash
python src/train.py --model-type cnn --run-name cnn_exp1 --epochs 10 --batch-size 16
```

**Train Path C (Pretrained WavLM fine-tuning):**
```bash
python src/train.py --model-type wavlm --run-name wavlm_exp1 --epochs 10 --batch-size 16
```

### 3.5 Standalone Evaluation
Evaluate any saved checkpoint on test data:
```bash
python src/evaluate.py --ckpt runs/cnn_exp1/best.pt --csv data/test.csv --out reports/cnn_test
```

### 3.6 Emotion Inference on New Audio
Predict emotion for a single audio file or directory:
```bash
python src/predict.py --ckpt runs/cnn_exp1/best.pt --input "path/to/speech.wav"
```
Supports sliding-window aggregation for long audio clips.
