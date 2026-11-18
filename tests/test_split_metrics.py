"""
test_split_metrics.py - Unit test suite for make_split.py and metrics.py.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.make_split import (
    EMOTION_CLASSES,
    check_leakage,
    split_fixed,
    split_kfold,
    split_stratified,
    verify_split_files,
)
from src.metrics import (
    compute_bootstrap_ci_by_subject,
    compute_ece,
    compute_metrics,
    restrict_labels,
    save_report,
)


@pytest.fixture
def mock_dataset_df():
    """Create a mock ESD-like manifest dataframe with 20 speakers and 5 classes."""
    records = []
    classes = ["neutral", "happy", "angry", "sad", "surprise"]
    for spk_idx in range(1, 21):
        spk_str = f"{spk_idx:04d}"
        lang = "zh" if spk_idx <= 10 else "en"
        for cls_idx, emotion in enumerate(classes):
            for utt in range(1, 6):  # 5 utterances per class
                records.append({
                    "path": f"D:/mock/{spk_str}/{emotion}/{spk_str}_{utt:06d}.wav",
                    "speaker": spk_str,
                    "lang": lang,
                    "emotion": emotion,
                    "label_id": cls_idx,
                    "utt_id": f"{utt:06d}",
                    "orig_split": "train",
                    "sr": 16000,
                    "dur": 3.0,
                    "md5": f"hash_{spk_str}_{emotion}_{utt}",
                })
    return pd.DataFrame(records)


def test_1_make_split_fixed_mode(mock_dataset_df):
    df_split = split_fixed(mock_dataset_df, train_range=(1, 14), val_range=(15, 16), test_range=(17, 20))
    train_spks = set(df_split[df_split["split"] == "train"]["speaker"])
    val_spks = set(df_split[df_split["split"] == "val"]["speaker"])
    test_spks = set(df_split[df_split["split"] == "test"]["speaker"])

    assert len(train_spks) == 14
    assert len(val_spks) == 2
    assert len(test_spks) == 4
    assert len(train_spks & val_spks) == 0
    assert len(train_spks & test_spks) == 0
    assert len(val_spks & test_spks) == 0


def test_2_make_split_stratified_mode(mock_dataset_df):
    df_split = split_stratified(mock_dataset_df, n_test_per_lang=2, n_val_per_lang=1, seed=42)
    val_df = df_split[df_split["split"] == "val"]
    test_df = df_split[df_split["split"] == "test"]

    # Val should have 1 zh and 1 en speaker
    assert val_df["speaker"].nunique() == 2
    assert set(val_df["lang"].unique()) == {"zh", "en"}

    # Test should have 2 zh and 2 en speakers
    assert test_df["speaker"].nunique() == 4
    assert set(test_df["lang"].unique()) == {"zh", "en"}
    assert test_df[test_df["lang"] == "zh"]["speaker"].nunique() == 2
    assert test_df[test_df["lang"] == "en"]["speaker"].nunique() == 2


def test_3_make_split_kfold_mode(mock_dataset_df):
    folds = split_kfold(mock_dataset_df, k=5, seed=42)
    assert len(folds) == 5

    tested_speakers = []
    for fold_df, info in folds:
        test_df = fold_df[fold_df["split"] == "test"]
        val_df = fold_df[fold_df["split"] == "val"]
        train_df = fold_df[fold_df["split"] == "train"]

        t_spks = set(test_df["speaker"])
        v_spks = set(val_df["speaker"])
        tr_spks = set(train_df["speaker"])

        assert len(t_spks & v_spks) == 0
        assert len(t_spks & tr_spks) == 0
        assert len(v_spks & tr_spks) == 0

        tested_speakers.extend(list(t_spks))

    # All 20 speakers tested exactly once
    assert len(tested_speakers) == 20
    assert len(set(tested_speakers)) == 20


def test_4_check_leakage_passes_clean_split(mock_dataset_df):
    df_split = split_stratified(mock_dataset_df, seed=42)
    train_df = df_split[df_split["split"] == "train"]
    val_df = df_split[df_split["split"] == "val"]
    test_df = df_split[df_split["split"] == "test"]

    is_valid, errors, _ = check_leakage(train_df, val_df, test_df, check_hash=True)
    assert is_valid is True
    assert len(errors) == 0


def test_5_check_leakage_detects_speaker_leakage(mock_dataset_df):
    df_split = split_stratified(mock_dataset_df, seed=42)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "val"].copy()
    test_df = df_split[df_split["split"] == "test"].copy()

    # Artificially inject a speaker into test
    leaked_spk = train_df["speaker"].iloc[0]
    leak_row = train_df[train_df["speaker"] == leaked_spk].iloc[0:1].copy()
    test_df = pd.concat([test_df, leak_row])

    is_valid, errors, _ = check_leakage(train_df, val_df, test_df)
    assert is_valid is False
    assert any("Speaker leakage" in e for e in errors)


def test_6_check_leakage_detects_path_leakage(mock_dataset_df):
    df_split = split_stratified(mock_dataset_df, seed=42)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "val"].copy()
    test_df = df_split[df_split["split"] == "test"].copy()

    # Introduce duplicate path with fake speaker ID
    duplicate_row = train_df.iloc[0:1].copy()
    duplicate_row["speaker"] = "9999"
    val_df = pd.concat([val_df, duplicate_row])

    is_valid, errors, _ = check_leakage(train_df, val_df, test_df)
    assert is_valid is False
    assert any("Path duplicate" in e for e in errors)


def test_7_check_leakage_detects_empty_split(mock_dataset_df):
    df_split = split_stratified(mock_dataset_df, seed=42)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "val"].copy()
    empty_test = df_split[df_split["split"] == "test"].iloc[0:0].copy()

    is_valid, errors, _ = check_leakage(train_df, val_df, empty_test)
    assert is_valid is False
    assert any("Test split is empty" in e for e in errors)


def test_8_check_leakage_detects_missing_class(mock_dataset_df):
    df_split = split_stratified(mock_dataset_df, seed=42)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "val"].copy()
    test_df = df_split[df_split["split"] == "test"].copy()

    # Drop surprise class from test
    test_df = test_df[test_df["emotion"] != "surprise"]

    is_valid, errors, _ = check_leakage(train_df, val_df, test_df)
    assert is_valid is False
    assert any("Missing classes in 'test' split" in e for e in errors)


def test_9_check_leakage_warns_language_absence(mock_dataset_df):
    df_split = split_fixed(mock_dataset_df, train_range=(1, 15), val_range=(16, 17), test_range=(18, 20))
    train_df = df_split[df_split["split"] == "train"]
    val_df = df_split[df_split["split"] == "val"]
    test_df = df_split[df_split["split"] == "test"]

    # Fixed mode has test range 18..20 (only 'en')
    is_valid, errors, warnings = check_leakage(train_df, val_df, test_df)
    assert is_valid is True
    assert any("Split 'test' only contains language(s) ['en']" in w for w in warnings)


def test_10_check_leakage_warns_class_imbalance(mock_dataset_df):
    df_split = split_stratified(mock_dataset_df, seed=42)
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "val"].copy()
    test_df = df_split[df_split["split"] == "test"].copy()

    # Artificially heavily bias test set towards neutral
    extra_neutral = test_df[test_df["emotion"] == "neutral"]
    test_df = pd.concat([test_df, extra_neutral, extra_neutral])

    is_valid, errors, warnings = check_leakage(train_df, val_df, test_df)
    assert any("Class imbalance" in w for w in warnings)


def test_11_compute_metrics_perfect_predictions():
    y_true = [0, 1, 2, 3, 4] * 10
    y_pred = list(y_true)
    spk = ["0001"] * 25 + ["0002"] * 25
    lng = ["zh"] * 25 + ["en"] * 25

    metrics = compute_metrics(y_true, y_pred, EMOTION_CLASSES, speakers=spk, langs=lng)
    assert metrics["uar"] == 1.0
    assert metrics["war"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["kappa"] == 1.0


def test_12_compute_metrics_random_predictions():
    np.random.seed(42)
    y_true = np.random.randint(0, 5, size=100)
    y_prob = np.random.dirichlet(np.ones(5), size=100)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_metrics(y_true, y_pred, EMOTION_CLASSES, y_prob=y_prob)
    assert 0.0 <= metrics["uar"] <= 1.0
    assert 0.0 <= metrics["war"] <= 1.0
    assert metrics["log_loss"] is not None and metrics["log_loss"] > 0
    assert metrics["top2_accuracy"] is not None
    assert 0.0 <= metrics["ece"] <= 1.0


def test_13_compute_ece():
    y_true = np.array([0, 1, 2, 3, 4])
    # Perfectly confident and 100% correct
    y_prob = np.eye(5)
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert pytest.approx(ece, abs=1e-5) == 0.0


def test_14_bootstrap_ci_by_subject():
    df = pd.DataFrame({
        "y_true": [0, 1, 2, 3, 4] * 20,
        "y_pred": [0, 1, 2, 3, 4] * 20,
        "speaker": ["0001"] * 50 + ["0002"] * 50,
    })
    ci = compute_bootstrap_ci_by_subject(df, EMOTION_CLASSES, n_bootstraps=50, seed=42)
    assert "uar" in ci
    assert "accuracy" in ci
    assert "macro_f1" in ci
    assert ci["uar"]["ci_lower"] <= ci["uar"]["mean"] <= ci["uar"]["ci_upper"]


def test_15_restrict_labels_cross_corpus():
    orig_classes = ["neutral", "happy", "angry", "sad", "surprise"]
    keep_classes = ["neutral", "happy", "angry", "sad"]

    y_true = ["neutral", "happy", "surprise", "angry"]
    y_prob = np.array([
        [0.4, 0.3, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.1, 0.1, 0.1],
        [0.0, 0.1, 0.1, 0.1, 0.7],  # surprise
        [0.1, 0.1, 0.7, 0.1, 0.0],
    ])

    yt_f, yp_f, prob_f, names = restrict_labels(y_true, y_prob, orig_classes, keep=keep_classes)

    assert len(yt_f) == 3  # surprise excluded
    assert names == keep_classes
    assert prob_f.shape == (3, 4)
    # Probabilities must sum to 1.0 per row
    assert np.allclose(prob_f.sum(axis=1), 1.0)


def test_16_save_report_generates_all_files(tmp_path):
    y_true = [0, 1, 2, 3, 4] * 10
    y_pred = [0, 1, 2, 3, 4] * 8 + [0] * 5 + [1] * 5  # Some mistakes (length 50)
    spk = ["0001"] * 25 + ["0002"] * 25
    lng = ["zh"] * 25 + ["en"] * 25
    y_prob = np.zeros((50, 5))
    for i, p in enumerate(y_pred):
        y_prob[i, p] = 0.8
        y_prob[i, (p + 1) % 5] = 0.2

    metrics = compute_metrics(y_true, y_pred, EMOTION_CLASSES, y_prob=y_prob, speakers=spk, langs=lng, n_bootstrap=20)
    save_report(
        metrics,
        out_dir=tmp_path,
        y_true=y_true,
        y_pred=y_pred,
        class_names=EMOTION_CLASSES,
        y_prob=y_prob,
        speakers=spk,
        langs=lng,
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.txt").exists()
    assert (tmp_path / "per_class.csv").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "top_confusions.csv").exists()
    assert (tmp_path / "per_speaker.csv").exists()
    assert (tmp_path / "per_lang.csv").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "predictions_annotated.csv").exists()
