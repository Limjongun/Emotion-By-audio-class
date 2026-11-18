"""
metrics.py - Comprehensive evaluation module for Speech Emotion Recognition (SER).

Metrics computed:
  - UAR (Unweighted Average Recall / Balanced Accuracy) [PRIMARY SER METRIC]
  - WAR (Weighted Average Recall / Standard Accuracy)
  - Macro-F1, Weighted-F1, Micro-F1, Macro-Precision, Macro-Recall
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa
  - Expected Calibration Error (ECE)
  - Log Loss & Multiclass ROC-AUC (OvR macro)
  - Top-2 Accuracy
  - Per-class metrics: Precision, Recall, F1, Support
  - Confusion Matrix (raw and normalized) & Top Confusion Pairs
  - Per-speaker breakdown (Accuracy, UAR, F1)
  - Per-language breakdown (zh vs en)
  - Subject-level (Speaker-level) 95% Bootstrap Confidence Intervals
  - Cross-corpus label restriction helper (restrict_labels)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

CANONICAL_CLASSES = ["neutral", "happy", "angry", "sad", "surprise"]


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures the weighted average difference between confidence and observed accuracy.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * (np.sum(in_bin) / n_samples)

    return float(ece)


def compute_bootstrap_ci_by_subject(
    df: pd.DataFrame,
    class_names: List[str],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Subject-level (Speaker-level) bootstrap confidence interval.
    Resamples unique speakers with replacement to capture true inter-speaker variance.
    """
    if "speaker" not in df.columns or df["speaker"].nunique() < 2:
        return {}

    rng = np.random.RandomState(seed)
    unique_speakers = df["speaker"].unique()
    n_spk = len(unique_speakers)

    uar_list: List[float] = []
    acc_list: List[float] = []
    f1_list: List[float] = []

    # Map labels if needed
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_true_all = df["y_true"].map(lambda x: label_to_idx.get(x, x)).to_numpy(dtype=int)
    y_pred_all = df["y_pred"].map(lambda x: label_to_idx.get(x, x)).to_numpy(dtype=int)

    # Group row indices by speaker for speed
    speaker_indices: Dict[Any, np.ndarray] = {
        spk: df.index[df["speaker"] == spk].to_numpy() for spk in unique_speakers
    }

    for _ in range(n_bootstraps):
        sampled_spks = rng.choice(unique_speakers, size=n_spk, replace=True)
        sample_row_idx = np.concatenate([speaker_indices[s] for s in sampled_spks])

        y_t = y_true_all[sample_row_idx]
        y_p = y_pred_all[sample_row_idx]

        # Calculate metrics on resampled subjects
        acc_list.append(float(accuracy_score(y_t, y_p)))
        uar_list.append(float(balanced_accuracy_score(y_t, y_p)))
        f1_list.append(float(f1_score(y_t, y_p, average="macro", zero_division=0)))

    alpha = (1.0 - ci_level) / 2.0
    low_pct = alpha * 100.0
    high_pct = (1.0 - alpha) * 100.0

    return {
        "uar": {
            "mean": float(np.mean(uar_list)),
            "std": float(np.std(uar_list)),
            "ci_lower": float(np.percentile(uar_list, low_pct)),
            "ci_upper": float(np.percentile(uar_list, high_pct)),
        },
        "accuracy": {
            "mean": float(np.mean(acc_list)),
            "std": float(np.std(acc_list)),
            "ci_lower": float(np.percentile(acc_list, low_pct)),
            "ci_upper": float(np.percentile(acc_list, high_pct)),
        },
        "macro_f1": {
            "mean": float(np.mean(f1_list)),
            "std": float(np.std(f1_list)),
            "ci_lower": float(np.percentile(f1_list, low_pct)),
            "ci_upper": float(np.percentile(f1_list, high_pct)),
        },
    }


def compute_metrics(
    y_true: Union[np.ndarray, List[Any]],
    y_pred: Union[np.ndarray, List[Any]],
    class_names: List[str] = CANONICAL_CLASSES,
    y_prob: Optional[np.ndarray] = None,
    speakers: Optional[Sequence[str]] = None,
    langs: Optional[Sequence[str]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for classification evaluation.
    """
    label_map = {name: idx for idx, name in enumerate(class_names)}
    n_classes = len(class_names)

    # Convert strings to integer indices if necessary
    def to_int_labels(arr: Union[np.ndarray, List[Any]]) -> np.ndarray:
        out = []
        for x in arr:
            if isinstance(x, (int, np.integer)):
                out.append(int(x))
            elif isinstance(x, str):
                if x in label_map:
                    out.append(label_map[x])
                elif x.lower() in label_map:
                    out.append(label_map[x.lower()])
                else:
                    raise ValueError(f"Unknown label string: '{x}'")
            else:
                out.append(int(x))
        return np.array(out, dtype=int)

    yt = to_int_labels(y_true)
    yp = to_int_labels(y_pred)

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true ({len(yt)}) vs y_pred ({len(yp)})")

    # 1. Global Metrics
    uar = float(balanced_accuracy_score(yt, yp))
    war = float(accuracy_score(yt, yp))
    macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(yt, yp, average="weighted", zero_division=0))
    macro_prec = float(precision_score(yt, yp, average="macro", zero_division=0))
    macro_rec = float(recall_score(yt, yp, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(yt, yp))
    kappa = float(cohen_kappa_score(yt, yp))

    # 2. Per-class metrics
    p, r, f, s = precision_recall_fscore_support(
        yt, yp, labels=list(range(n_classes)), zero_division=0
    )
    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f[idx]),
            "support": int(s[idx]),
        }

    # 3. Confusion Matrix
    cm = confusion_matrix(yt, yp, labels=list(range(n_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # Top confusions (excluding correct diagonal)
    conf_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                conf_pairs.append({
                    "true_class": class_names[i],
                    "pred_class": class_names[j],
                    "count": int(cm[i, j]),
                    "rate_pct": float(round(cm_norm[i, j] * 100.0, 2)),
                })
    conf_pairs.sort(key=lambda x: x["count"], reverse=True)
    top_confusions = conf_pairs[:10]

    # 4. Probability-based metrics (if y_prob provided)
    logloss_val = None
    ece_val = None
    top2_acc = None
    roc_auc_val = None

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.shape == (len(yt), n_classes):
            try:
                logloss_val = float(log_loss(yt, y_prob, labels=list(range(n_classes))))
            except Exception:
                pass
            
            ece_val = compute_ece(yt, y_prob)

            # Top-2 accuracy
            top2_preds = np.argsort(y_prob, axis=1)[:, -2:]
            top2_hits = np.any(top2_preds == yt[:, None], axis=1)
            top2_acc = float(np.mean(top2_hits))

            # Multiclass ROC-AUC (OvR macro)
            try:
                # Only if all classes are present in yt
                if len(np.unique(yt)) == n_classes:
                    roc_auc_val = float(roc_auc_score(yt, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                pass

    # 5. Per-speaker & Per-language breakdowns
    per_speaker: Dict[str, Dict[str, float]] = {}
    if speakers is not None:
        spk_series = pd.Series(speakers)
        for spk in sorted(spk_series.unique()):
            idx = np.where(spk_series == spk)[0]
            if len(idx) > 0:
                y_t_spk = yt[idx]
                y_p_spk = yp[idx]
                per_speaker[str(spk)] = {
                    "accuracy": float(accuracy_score(y_t_spk, y_p_spk)),
                    "uar": float(balanced_accuracy_score(y_t_spk, y_p_spk)),
                    "macro_f1": float(f1_score(y_t_spk, y_p_spk, average="macro", zero_division=0)),
                    "support": int(len(idx)),
                }

    per_lang: Dict[str, Dict[str, float]] = {}
    if langs is not None:
        lang_series = pd.Series(langs)
        for lg in sorted(lang_series.unique()):
            idx = np.where(lang_series == lg)[0]
            if len(idx) > 0:
                y_t_lg = yt[idx]
                y_p_lg = yp[idx]
                per_lang[str(lg)] = {
                    "accuracy": float(accuracy_score(y_t_lg, y_p_lg)),
                    "uar": float(balanced_accuracy_score(y_t_lg, y_p_lg)),
                    "macro_f1": float(f1_score(y_t_lg, y_p_lg, average="macro", zero_division=0)),
                    "support": int(len(idx)),
                }

    # 6. Subject-level bootstrap CI
    bootstrap_ci = {}
    if speakers is not None:
        df_eval = pd.DataFrame({"y_true": yt, "y_pred": yp, "speaker": list(speakers)})
        bootstrap_ci = compute_bootstrap_ci_by_subject(
            df_eval, class_names, n_bootstraps=n_bootstrap, seed=seed
        )

    return {
        "uar": uar,
        "war": war,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "mcc": mcc,
        "kappa": kappa,
        "log_loss": logloss_val,
        "ece": ece_val,
        "top2_accuracy": top2_acc,
        "roc_auc_ovr": roc_auc_val,
        "per_class": per_class,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "top_confusions": top_confusions,
        "per_speaker": per_speaker,
        "per_lang": per_lang,
        "bootstrap_ci": bootstrap_ci,
        "n_samples": int(len(yt)),
        "class_names": class_names,
    }


def restrict_labels(
    y_true: Union[np.ndarray, List[Any]],
    y_prob: np.ndarray,
    original_classes: List[str],
    keep: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Filter predictions and renormalize probabilities to a subset of classes.
    Used for Cross-Corpus evaluation (e.g. evaluating an ESD model on CREMA-D 4-class).
    
    Returns:
      (filtered_y_true, filtered_y_pred, filtered_y_prob, new_class_names)
    """
    keep_indices = [original_classes.index(c) for c in keep]
    label_map_orig = {c: i for i, c in enumerate(original_classes)}

    # Convert true labels to string if needed
    y_true_str = []
    for y in y_true:
        if isinstance(y, (int, np.integer)):
            y_true_str.append(original_classes[int(y)])
        else:
            y_true_str.append(str(y).lower())

    mask = [lbl in keep for lbl in y_true_str]
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(f"No samples matched keep classes: {keep}")

    filtered_y_true = np.array([keep.index(y_true_str[i]) for i in indices], dtype=int)
    sliced_prob = y_prob[indices][:, keep_indices]

    # Renormalize probabilities across the kept classes
    row_sums = sliced_prob.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        filtered_y_prob = np.where(row_sums > 0, sliced_prob / row_sums, 1.0 / len(keep))

    filtered_y_pred = np.argmax(filtered_y_prob, axis=1)

    return filtered_y_true, filtered_y_pred, filtered_y_prob, keep


def save_report(
    metrics: Dict[str, Any],
    out_dir: str | Path,
    y_true: Optional[Sequence[Any]] = None,
    y_pred: Optional[Sequence[Any]] = None,
    class_names: Optional[List[str]] = None,
    y_prob: Optional[np.ndarray] = None,
    paths: Optional[Sequence[str]] = None,
    speakers: Optional[Sequence[str]] = None,
    langs: Optional[Sequence[str]] = None,
    save_plot: bool = True,
) -> str:
    """
    Save all evaluation artifacts to out_dir:
      - summary.json
      - summary.txt
      - per_class.csv
      - confusion_matrix.csv
      - top_confusions.csv
      - per_speaker.csv (if available)
      - per_lang.csv (if available)
      - confusion_matrix.png (if matplotlib available)
      - predictions_annotated.csv (if y_true & y_pred passed)
    """
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cls_names = class_names or metrics.get("class_names", CANONICAL_CLASSES)

    # 1. Save summary.json
    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. Build summary.txt
    lines = [
        "=" * 60,
        "SPEECH EMOTION RECOGNITION (SER) EVALUATION REPORT",
        "=" * 60,
        f"Total Samples: {metrics['n_samples']}",
        f"Primary Metric (UAR / Balanced Accuracy): {metrics['uar']*100:.2f}%",
        f"Weighted Accuracy (WAR):                   {metrics['war']*100:.2f}%",
        f"Macro F1-Score:                            {metrics['macro_f1']*100:.2f}%",
        f"Matthews Correlation Coeff (MCC):          {metrics['mcc']:.4f}",
        f"Cohen's Kappa:                             {metrics['kappa']:.4f}",
    ]

    if metrics.get("ece") is not None:
        lines.append(f"Expected Calibration Error (ECE):          {metrics['ece']:.4f}")
    if metrics.get("log_loss") is not None:
        lines.append(f"Log Loss (Cross-Entropy):                  {metrics['log_loss']:.4f}")
    if metrics.get("top2_accuracy") is not None:
        lines.append(f"Top-2 Accuracy:                            {metrics['top2_accuracy']*100:.2f}%")
    if metrics.get("roc_auc_ovr") is not None:
        lines.append(f"ROC-AUC (OvR Macro):                       {metrics['roc_auc_ovr']:.4f}")

    # Bootstrap CI
    ci = metrics.get("bootstrap_ci", {})
    if ci:
        lines.append("-" * 60)
        lines.append("Subject-Level 95% Bootstrap Confidence Intervals (Speaker Resampled):")
        for k in ["uar", "accuracy", "macro_f1"]:
            if k in ci:
                c = ci[k]
                lines.append(f"  {k.upper():<10}: {c['mean']*100:.2f}% ± {c['std']*100:.2f}%  [95% CI: {c['ci_lower']*100:.2f}% - {c['ci_upper']*100:.2f}%]")

    # Per class summary
    lines.append("-" * 60)
    lines.append(f"{'Class':<12} {'Precision (%)':<14} {'Recall (%)':<12} {'F1 (%)':<10} {'Support':<8}")
    lines.append("-" * 60)
    for name in cls_names:
        c_stat = metrics["per_class"].get(name, {})
        lines.append(
            f"{name:<12} {c_stat.get('precision', 0.0)*100:<14.2f} "
            f"{c_stat.get('recall', 0.0)*100:<12.2f} "
            f"{c_stat.get('f1', 0.0)*100:<10.2f} "
            f"{c_stat.get('support', 0):<8}"
        )

    # Top confusions
    if metrics.get("top_confusions"):
        lines.append("-" * 60)
        lines.append("Top Confusion Pairs (Most Frequently Confused):")
        for item in metrics["top_confusions"][:5]:
            lines.append(f"  True: {item['true_class']:<10} -> Pred: {item['pred_class']:<10} (Count: {item['count']}, {item['rate_pct']}%)")

    # Languages
    if metrics.get("per_lang"):
        lines.append("-" * 60)
        lines.append("Breakdown by Language:")
        for lg, st in metrics["per_lang"].items():
            lines.append(f"  Lang {lg}: Acc={st['accuracy']*100:.2f}%, UAR={st['uar']*100:.2f}%, F1={st['macro_f1']*100:.2f}% (N={st['support']})")

    lines.append("=" * 60)
    summary_text = "\n".join(lines)
    with open(out_path / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # 3. Save per_class.csv
    per_class_rows = []
    for name in cls_names:
        row = {"class": name}
        row.update(metrics["per_class"].get(name, {}))
        per_class_rows.append(row)
    pd.DataFrame(per_class_rows).to_csv(out_path / "per_class.csv", index=False)

    # 4. Save confusion_matrix.csv
    cm_df = pd.DataFrame(metrics["confusion_matrix_raw"], index=cls_names, columns=cls_names)
    cm_df.to_csv(out_path / "confusion_matrix.csv")

    # 5. Save top_confusions.csv
    if metrics.get("top_confusions"):
        pd.DataFrame(metrics["top_confusions"]).to_csv(out_path / "top_confusions.csv", index=False)

    # 6. Save per_speaker.csv
    if metrics.get("per_speaker"):
        spk_rows = [{"speaker": k, **v} for k, v in metrics["per_speaker"].items()]
        pd.DataFrame(spk_rows).to_csv(out_path / "per_speaker.csv", index=False)

    # 7. Save per_lang.csv
    if metrics.get("per_lang"):
        lang_rows = [{"lang": k, **v} for k, v in metrics["per_lang"].items()]
        pd.DataFrame(lang_rows).to_csv(out_path / "per_lang.csv", index=False)

    # 8. Save confusion matrix image
    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            cm_norm = np.array(metrics["confusion_matrix_norm"])
            im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set(
                xticks=np.arange(len(cls_names)),
                yticks=np.arange(len(cls_names)),
                xticklabels=cls_names,
                yticklabels=cls_names,
                ylabel="True Label",
                xlabel="Predicted Label",
                title=f"Normalized Confusion Matrix (UAR: {metrics['uar']*100:.1f}%)",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            thresh = cm_norm.max() / 2.0
            for i in range(len(cls_names)):
                for j in range(len(cls_names)):
                    cnt = metrics["confusion_matrix_raw"][i][j]
                    pct = cm_norm[i, j] * 100.0
                    ax.text(
                        j, i, f"{pct:.1f}%\n({cnt})",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=8,
                    )
            plt.tight_layout()
            plt.savefig(out_path / "confusion_matrix.png", dpi=200)
            plt.close(fig)
        except Exception as e:
            print(f"[!] Warning: failed to save confusion matrix plot: {e}")

    # 9. Annotated predictions CSV (if true and pred provided)
    if y_true is not None and y_pred is not None:
        pred_dict: Dict[str, Any] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "is_correct": [yt == yp for yt, yp in zip(y_true, y_pred)],
        }
        if paths is not None:
            pred_dict["path"] = paths
        if speakers is not None:
            pred_dict["speaker"] = speakers
        if langs is not None:
            pred_dict["lang"] = langs
        if y_prob is not None:
            for idx, c in enumerate(cls_names):
                pred_dict[f"p_{c}"] = y_prob[:, idx]

        pd.DataFrame(pred_dict).to_csv(out_path / "predictions_annotated.csv", index=False)

    return summary_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SER predictions from CSV file")
    parser.add_argument("--pred", type=str, required=True, help="Path to predictions.csv")
    parser.add_argument("--out", type=str, default="reports/latest", help="Output directory for reports")
    parser.add_argument("--classes", nargs="+", default=CANONICAL_CLASSES, help="Class names list")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations for CI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred, dtype={"speaker": str})
    if "y_true" not in pred_df.columns or "y_pred" not in pred_df.columns:
        raise ValueError("Input CSV must contain 'y_true' and 'y_pred' columns.")

    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    speakers = pred_df["speaker"].values if "speaker" in pred_df.columns else None
    langs = pred_df["lang"].values if "lang" in pred_df.columns else None
    paths = pred_df["path"].values if "path" in pred_df.columns else None

    # Check for probability columns p_<class>
    prob_cols = [f"p_{c}" for c in args.classes]
    y_prob = None
    if all(col in pred_df.columns for col in prob_cols):
        y_prob = pred_df[prob_cols].values

    metrics = compute_metrics(
        y_true,
        y_pred,
        class_names=args.classes,
        y_prob=y_prob,
        speakers=speakers,
        langs=langs,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    report_text = save_report(
        metrics,
        out_dir=args.out,
        y_true=y_true,
        y_pred=y_pred,
        class_names=args.classes,
        y_prob=y_prob,
        paths=paths,
        speakers=speakers,
        langs=langs,
    )

    print(report_text)
    print(f"\n[+] Full reports and artifacts saved to: {args.out}")


if __name__ == "__main__":
    main()
