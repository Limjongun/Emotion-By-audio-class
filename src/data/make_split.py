"""
make_split.py - Manifest generation, speaker-independent splitting, and leakage verification.

Supports three splitting modes:
  1. fixed: User-specified speaker ID ranges (e.g. train 1-15, val 16-17, test 18-20)
  2. stratified: Balanced speaker sampling per language (e.g. 14 train / 2 val / 4 test)
  3. kfold: Group K-Fold over speakers (each speaker appears in test exactly once)

Includes comprehensive anti-leakage checks (speaker isolation, duplicate files/hashes,
class representation, and language coverage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Canonical emotion classes and mapping
EMOTION_CLASSES = ["neutral", "happy", "angry", "sad", "surprise"]
LABEL_MAP = {cls: idx for idx, cls in enumerate(EMOTION_CLASSES)}


def get_audio_info_wave(path: Path | str) -> Tuple[int, float]:
    """
    Get sample rate and duration in seconds using standard library wave module.
    Falls back to soundfile if wave fails (e.g. for non-standard WAV headers).
    """
    path_str = str(path)
    try:
        with wave.open(path_str, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            dur = n_frames / float(sr) if sr > 0 else 0.0
            return sr, round(dur, 4)
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(path_str)
            return info.samplerate, round(info.duration, 4)
        except Exception as e:
            raise RuntimeError(f"Failed to read WAV info from {path}: {e}")


def compute_file_hash(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute MD5 hash of an audio file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_dataset(
    root_dir: str | Path,
    zh_max_id: int = 10,
    compute_hash: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan ESD dataset folder recursively and build a manifest DataFrame.
    
    Expected folder structure:
      root_dir/
        <speaker_id>/
          <Emotion>/
            [optional subfolders like train/evaluation/test]
              *.wav
    """
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    records: List[Dict[str, Any]] = []
    skipped_files: List[str] = []

    if verbose:
        print(f"[*] Scanning dataset at: {root}")

    # Find all .wav files recursively
    wav_files = sorted(root.glob("**/*.wav"))
    if not wav_files:
        raise ValueError(f"No .wav files found in {root}")

    for wav_path in wav_files:
        rel_parts = wav_path.relative_to(root).parts
        if len(rel_parts) < 2:
            skipped_files.append(str(wav_path))
            continue

        spk_id = rel_parts[0]
        # Check if spk_id is valid speaker format (e.g. '0001')
        spk_num = None
        try:
            spk_num = int(spk_id)
        except ValueError:
            skipped_files.append(str(wav_path))
            continue

        spk_str = f"{spk_num:04d}"

        # Emotion folder is typically level 1 (under speaker)
        raw_emotion = rel_parts[1].lower()
        if raw_emotion not in LABEL_MAP:
            # Maybe inside subfolders, find any part matching canonical emotion
            matched = [p.lower() for p in rel_parts if p.lower() in LABEL_MAP]
            if matched:
                raw_emotion = matched[0]
            else:
                skipped_files.append(str(wav_path))
                continue

        emotion = raw_emotion
        label_id = LABEL_MAP[emotion]

        # Language heuristic: 1..zh_max_id -> zh, else en
        lang = "zh" if spk_num <= zh_max_id else "en"

        # Check for original split subfolders (train/evaluation/test)
        orig_split = ""
        for p in rel_parts:
            low = p.lower()
            if low in ("train", "evaluation", "test"):
                orig_split = low
                break

        # Utterance ID from stem (e.g. 0001_000351 -> 000351 or full stem)
        stem = wav_path.stem
        utt_id = stem.split("_")[-1] if "_" in stem else stem

        try:
            sr, dur = get_audio_info_wave(wav_path)
        except Exception as e:
            if verbose:
                print(f"[!] Warning: skipped corrupt file {wav_path}: {e}")
            skipped_files.append(str(wav_path))
            continue

        md5 = compute_file_hash(wav_path) if compute_hash else ""

        records.append({
            "path": str(wav_path).replace("\\", "/"),
            "speaker": spk_str,
            "lang": lang,
            "emotion": emotion,
            "label_id": label_id,
            "utt_id": utt_id,
            "orig_split": orig_split,
            "sr": sr,
            "dur": dur,
            "md5": md5,
        })

    if verbose and skipped_files:
        print(f"[*] Skipped {len(skipped_files)} invalid or non-ESD files.")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"Scan completed but 0 valid audio records found in {root}")

    if verbose:
        print(f"[*] Total valid utterances: {len(df)}")
        print(f"[*] Speakers found ({len(df['speaker'].unique())}): {sorted(df['speaker'].unique())}")
        print(f"[*] Languages: {df['lang'].value_counts().to_dict()}")
        print(f"[*] Emotion counts:\n{df['emotion'].value_counts().to_string()}")

    return df


def split_fixed(
    df: pd.DataFrame,
    train_range: Tuple[int, int] = (1, 15),
    val_range: Tuple[int, int] = (16, 17),
    test_range: Tuple[int, int] = (18, 20),
) -> pd.DataFrame:
    """Assign splits based on fixed speaker integer ID ranges."""
    train_spks = {f"{i:04d}" for i in range(train_range[0], train_range[1] + 1)}
    val_spks = {f"{i:04d}" for i in range(val_range[0], val_range[1] + 1)}
    test_spks = {f"{i:04d}" for i in range(test_range[0], test_range[1] + 1)}

    # Ensure no range overlap
    if train_spks & val_spks:
        raise ValueError(f"Train and Val ranges overlap: {train_spks & val_spks}")
    if train_spks & test_spks:
        raise ValueError(f"Train and Test ranges overlap: {train_spks & test_spks}")
    if val_spks & test_spks:
        raise ValueError(f"Val and Test ranges overlap: {val_spks & test_spks}")

    df = df.copy()

    def get_split(spk: str) -> str:
        if spk in train_spks:
            return "train"
        if spk in val_spks:
            return "val"
        if spk in test_spks:
            return "test"
        return "unassigned"

    df["split"] = df["speaker"].apply(get_split)
    unassigned = df[df["split"] == "unassigned"]
    if not unassigned.empty:
        print(f"[!] Warning: {len(unassigned)} utterances from speakers {sorted(unassigned['speaker'].unique())} are unassigned.")
        df = df[df["split"] != "unassigned"].copy()

    return df


def split_stratified(
    df: pd.DataFrame,
    n_test_per_lang: int = 2,
    n_val_per_lang: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign splits speaker-independently with balanced language representation.
    Randomly chooses test and val speakers from each language group.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()

    train_spks: Set[str] = set()
    val_spks: Set[str] = set()
    test_spks: Set[str] = set()

    for lang in sorted(df["lang"].unique()):
        spks = sorted(df[df["lang"] == lang]["speaker"].unique())
        shuffled = list(spks)
        rng.shuffle(shuffled)

        if len(shuffled) < n_test_per_lang + n_val_per_lang + 1:
            raise ValueError(
                f"Not enough speakers for lang '{lang}': found {len(shuffled)}, "
                f"need at least {n_test_per_lang + n_val_per_lang + 1}."
            )

        test_part = shuffled[:n_test_per_lang]
        val_part = shuffled[n_test_per_lang : n_test_per_lang + n_val_per_lang]
        train_part = shuffled[n_test_per_lang + n_val_per_lang :]

        test_spks.update(test_part)
        val_spks.update(val_part)
        train_spks.update(train_part)

    def get_split(spk: str) -> str:
        if spk in train_spks:
            return "train"
        if spk in val_spks:
            return "val"
        if spk in test_spks:
            return "test"
        raise RuntimeError(f"Speaker {spk} not allocated.")

    df["split"] = df["speaker"].apply(get_split)
    return df


def split_kfold(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Group K-Fold per speaker.
    Returns a list of k split DataFrames, where each speaker appears in the test set
    of exactly one fold.
    """
    rng = np.random.RandomState(seed)
    speakers = sorted(df["speaker"].unique())
    n_speakers = len(speakers)

    if n_speakers < k:
        raise ValueError(f"Cannot create {k} folds with only {n_speakers} speakers.")

    # Group speakers by language for balanced distribution across folds
    folds_spk: List[List[str]] = [[] for _ in range(k)]
    for lang in sorted(df["lang"].unique()):
        lang_spks = sorted(df[df["lang"] == lang]["speaker"].unique())
        shuffled = list(lang_spks)
        rng.shuffle(shuffled)
        for idx, spk in enumerate(shuffled):
            folds_spk[idx % k].append(spk)

    results: List[Tuple[pd.DataFrame, Dict[str, Any]]] = []
    all_test_speakers: List[str] = []

    for fold_idx in range(k):
        test_spks = set(folds_spk[fold_idx])
        val_spks = set(folds_spk[(fold_idx + 1) % k])
        train_spks = set(speakers) - test_spks - val_spks

        all_test_speakers.extend(test_spks)

        fold_df = df.copy()

        def get_split(spk: str) -> str:
            if spk in train_spks:
                return "train"
            if spk in val_spks:
                return "val"
            if spk in test_spks:
                return "test"
            raise RuntimeError(f"Speaker {spk} unassigned in fold {fold_idx}")

        fold_df["split"] = fold_df["speaker"].apply(get_split)

        info = {
            "fold": fold_idx,
            "train_speakers": sorted(train_spks),
            "val_speakers": sorted(val_spks),
            "test_speakers": sorted(test_spks),
        }
        results.append((fold_df, info))

    # Verification: Each speaker is in test exactly once
    assert len(all_test_speakers) == len(speakers), "Speaker count mismatch in K-Fold test sets"
    assert set(all_test_speakers) == set(speakers), "Some speakers were not tested or tested multiple times"

    return results


def check_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    check_hash: bool = False,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """
    Verify anti-leakage constraints between splits.
    
    Returns:
      (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    train_spks = set(train_df["speaker"].astype(str).unique())
    val_spks = set(val_df["speaker"].astype(str).unique())
    test_spks = set(test_df["speaker"].astype(str).unique())

    # 1. Speaker isolation check (CRITICAL)
    train_val_spk = train_spks & val_spks
    if train_val_spk:
        errors.append(f"Speaker leakage between Train and Val: {sorted(train_val_spk)}")

    train_test_spk = train_spks & test_spks
    if train_test_spk:
        errors.append(f"Speaker leakage between Train and Test: {sorted(train_test_spk)}")

    val_test_spk = val_spks & test_spks
    if val_test_spk:
        errors.append(f"Speaker leakage between Val and Test: {sorted(val_test_spk)}")

    # 2. Path duplication check
    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    test_paths = set(test_df["path"])

    if train_paths & val_paths:
        errors.append(f"Path duplicate between Train and Val: {len(train_paths & val_paths)} files")
    if train_paths & test_paths:
        errors.append(f"Path duplicate between Train and Test: {len(train_paths & test_paths)} files")
    if val_paths & test_paths:
        errors.append(f"Path duplicate between Val and Test: {len(val_paths & test_paths)} files")

    # 3. Hash duplication check (optional)
    if check_hash:
        has_hash = "md5" in train_df.columns and (train_df["md5"] != "").all()
        if has_hash:
            train_hashes = set(train_df["md5"])
            val_hashes = set(val_df["md5"])
            test_hashes = set(test_df["md5"])
            if train_hashes & val_hashes:
                errors.append(f"MD5 hash duplicate between Train and Val: {len(train_hashes & val_hashes)} files")
            if train_hashes & test_hashes:
                errors.append(f"MD5 hash duplicate between Train and Test: {len(train_hashes & test_hashes)} files")
            if val_hashes & test_hashes:
                errors.append(f"MD5 hash duplicate between Val and Test: {len(val_hashes & test_hashes)} files")

    # 4. Non-empty split check
    if len(train_df) == 0:
        errors.append("Train split is empty!")
    if len(val_df) == 0:
        errors.append("Val split is empty!")
    if len(test_df) == 0:
        errors.append("Test split is empty!")

    # 5. Class completeness check (all 5 classes present in each split)
    expected_classes = set(EMOTION_CLASSES)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        present_classes = set(df["emotion"].unique())
        missing = expected_classes - present_classes
        if missing:
            errors.append(f"Missing classes in '{name}' split: {sorted(missing)}")

    # 6. Language coverage check (Warning)
    all_langs = set(train_df["lang"].unique()) | set(val_df["lang"].unique()) | set(test_df["lang"].unique())
    for name, df in [("val", val_df), ("test", test_df)]:
        present = set(df["lang"].unique())
        missing = all_langs - present
        if missing:
            warnings.append(
                f"Split '{name}' only contains language(s) {sorted(present)}; "
                f"language(s) {sorted(missing)} generalization is not measured."
            )

    # 7. Class balance ratio check (Warning)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = df["emotion"].value_counts()
        if len(counts) > 0:
            ratio = counts.max() / counts.min()
            if ratio > 1.5:
                warnings.append(
                    f"Class imbalance in '{name}' split: max/min ratio is {ratio:.2f} (> 1.5)."
                )

    # 8. Parallel utterance ID note
    train_utts = set(train_df["utt_id"])
    test_utts = set(test_df["utt_id"])
    overlap_utts = train_utts & test_utts
    if overlap_utts and verbose:
        print(f"[*] Note: {len(overlap_utts)} parallel text utterance IDs shared between train and test (expected in ESD).")

    is_valid = len(errors) == 0
    if strict and warnings:
        errors.extend([f"[Strict Mode Error] {w}" for w in warnings])
        is_valid = len(errors) == 0

    if verbose:
        if errors:
            print("[!] ANTI-LEAKAGE CHECK FAILED:")
            for e in errors:
                print(f"    - ERROR: {e}")
        else:
            print("[+] ANTI-LEAKAGE CHECK PASSED (No leakage errors)")

        if warnings:
            print("[!] WARNINGS:")
            for w in warnings:
                print(f"    - WARNING: {w}")

    return is_valid, errors, warnings


def verify_split_files(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    check_hash: bool = False,
    strict: bool = False,
    verbose: bool = True,
) -> bool:
    """Load train/val/test CSV files and verify anti-leakage constraints."""
    if verbose:
        print(f"[*] Verifying splits:\n  Train: {train_csv}\n  Val:   {val_csv}\n  Test:  {test_csv}")

    train_df = pd.read_csv(train_csv, dtype={"speaker": str, "utt_id": str})
    val_df = pd.read_csv(val_csv, dtype={"speaker": str, "utt_id": str})
    test_df = pd.read_csv(test_csv, dtype={"speaker": str, "utt_id": str})

    is_valid, errors, warnings = check_leakage(
        train_df, val_df, test_df,
        check_hash=check_hash, strict=strict, verbose=verbose
    )

    if not is_valid:
        raise ValueError(f"Anti-leakage validation failed with errors: {errors}")

    return True


def generate_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    errors: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    """Compile structured summary report of dataset split."""
    def split_stats(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "n_files": int(len(df)),
            "n_speakers": int(df["speaker"].nunique()),
            "speakers": sorted(df["speaker"].unique().tolist()),
            "total_duration_hours": round(float(df["dur"].sum()) / 3600.0, 3) if "dur" in df else 0.0,
            "avg_duration_sec": round(float(df["dur"].mean()), 2) if "dur" in df else 0.0,
            "languages": df["lang"].value_counts().to_dict(),
            "emotions": df["emotion"].value_counts().to_dict(),
        }

    return {
        "mode": mode,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "splits": {
            "train": split_stats(train_df),
            "val": split_stats(val_df),
            "test": split_stats(test_df),
        },
    }


def save_splits(
    manifest_df: pd.DataFrame,
    split_df: pd.DataFrame,
    out_dir: str | Path,
    mode: str,
    check_hash: bool = False,
    strict: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Save manifest.csv, split.csv, train.csv, val.csv, test.csv, and split_report.json."""
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    # Run anti-leakage checks
    is_valid, errors, warnings = check_leakage(
        train_df, val_df, test_df,
        check_hash=check_hash, strict=strict, verbose=verbose
    )

    if not is_valid:
        raise ValueError(f"Aborting save: Anti-leakage checks failed: {errors}")

    # Save manifest.csv
    manifest_csv = out_path / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    # Save split.csv
    split_csv = out_path / "split.csv"
    split_df.to_csv(split_csv, index=False)

    # Save individual split CSVs
    train_csv = out_path / "train.csv"
    val_csv = out_path / "val.csv"
    test_csv = out_path / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Generate and save report
    report = generate_split_report(train_df, val_df, test_df, mode, errors, warnings)
    report_json = out_path / "split_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"[+] Splits saved successfully to: {out_path}")
        print(f"  train: {len(train_df)} files | {train_df['speaker'].nunique()} speakers | langs {train_df['lang'].value_counts().to_dict()}")
        print(f"  val  : {len(val_df)} files | {val_df['speaker'].nunique()} speakers | langs {val_df['lang'].value_counts().to_dict()}")
        print(f"  test : {len(test_df)} files | {test_df['speaker'].nunique()} speakers | langs {test_df['lang'].value_counts().to_dict()}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ESD Dataset Manifest and Speaker-Independent Split Generator")
    parser.add_argument("--root", type=str, default="D:/datasets/archive/Emotion Speech Dataset", help="Dataset root directory")
    parser.add_argument("--out", type=str, default="data", help="Output directory for generated CSVs")
    parser.add_argument("--mode", type=str, choices=["fixed", "stratified", "kfold"], default="stratified", help="Split mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--zh-max-id", type=int, default=10, help="Max speaker ID for Mandarin speakers")
    parser.add_argument("--train", nargs=2, type=int, default=[1, 15], help="Fixed mode train speaker ID range [start, end]")
    parser.add_argument("--val", nargs=2, type=int, default=[16, 17], help="Fixed mode val speaker ID range [start, end]")
    parser.add_argument("--test", nargs=2, type=int, default=[18, 20], help="Fixed mode test speaker ID range [start, end]")
    parser.add_argument("--n-test-per-lang", type=int, default=2, help="Stratified mode: test speakers per language")
    parser.add_argument("--n-val-per-lang", type=int, default=1, help="Stratified mode: val speakers per language")
    parser.add_argument("--k", type=int, default=5, help="K-Fold mode: number of folds")
    parser.add_argument("--hash", action="store_true", help="Compute and check MD5 hashes")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--verify", type=str, default="", help="Directory containing train.csv, val.csv, test.csv to verify")

    args = parser.parse_args()

    # Standalone verification mode
    if args.verify:
        v_dir = Path(args.verify)
        verify_split_files(
            v_dir / "train.csv",
            v_dir / "val.csv",
            v_dir / "test.csv",
            check_hash=args.hash,
            strict=args.strict,
        )
        return

    # 1. Scan dataset and build manifest
    manifest_df = scan_dataset(
        args.root,
        zh_max_id=args.zh_max_id,
        compute_hash=args.hash,
        verbose=True,
    )

    out_dir = Path(args.out)

    # 2. Execute chosen split mode
    if args.mode == "fixed":
        split_df = split_fixed(
            manifest_df,
            train_range=(args.train[0], args.train[1]),
            val_range=(args.val[0], args.val[1]),
            test_range=(args.test[0], args.test[1]),
        )
        save_splits(manifest_df, split_df, out_dir, mode="fixed", check_hash=args.hash, strict=args.strict)

    elif args.mode == "stratified":
        split_df = split_stratified(
            manifest_df,
            n_test_per_lang=args.n_test_per_lang,
            n_val_per_lang=args.n_val_per_lang,
            seed=args.seed,
        )
        save_splits(manifest_df, split_df, out_dir, mode="stratified", check_hash=args.hash, strict=args.strict)

    elif args.mode == "kfold":
        folds = split_kfold(manifest_df, k=args.k, seed=args.seed)
        kfold_dir = out_dir / "kfold"
        kfold_dir.mkdir(parents=True, exist_ok=True)

        manifest_df.to_csv(out_dir / "manifest.csv", index=False)
        folds_meta = []

        for fold_df, info in folds:
            fold_idx = info["fold"]
            f_dir = kfold_dir / f"fold{fold_idx}"
            save_splits(manifest_df, fold_df, f_dir, mode=f"kfold-fold{fold_idx}", check_hash=args.hash, strict=args.strict, verbose=False)
            folds_meta.append(info)
            print(f"[+] Fold {fold_idx}: Train {len(info['train_speakers'])} spks, Val {len(info['val_speakers'])} spks, Test {len(info['test_speakers'])} spks")

        with open(kfold_dir / "folds.json", "w", encoding="utf-8") as f:
            json.dump(folds_meta, f, indent=2)

        # Default fold0 as root train/val/test for convenience
        fold0_df = folds[0][0]
        save_splits(manifest_df, fold0_df, out_dir, mode="kfold-fold0", check_hash=args.hash, strict=args.strict, verbose=True)


if __name__ == "__main__":
    main()
