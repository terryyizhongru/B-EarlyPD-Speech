#!/usr/bin/env python3
"""Aggregate recording-level predictions from a setting or one task directory."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = (
    "Accuracy",
    "F1",
    "Precision",
    "AUC",
    "Sensitivity",
    "Specificity",
    "BalancedAccuracy",
)
COHORTS = ("Combined", "PC-GITA", "NeuroVoz")


def numbered_dirs(root: Path, prefix: str) -> list[Path]:
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)$")
    found = []
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found)]


def resolve_task_dirs(input_dir: Path) -> tuple[Path, list[Path]]:
    """Accept either a setting root or one folds_tsv_* task directory."""
    if input_dir.name.startswith("folds_tsv_"):
        return input_dir.parent, [input_dir]
    tasks = sorted(
        path
        for path in input_dir.glob("folds_tsv_*")
        if path.is_dir() and "bak" not in path.name.lower()
    )
    return input_dir, tasks


def task_name(path: Path) -> str:
    name = path.name.upper()
    if "SENTENCES" in name:
        return "Sentence"
    if "SUSTAINED-VOWELS" in name:
        return "Sustained vowel /a/"
    if "DDK_ANALYSIS" in name:
        return "DDK /pa-ta-ka/"
    return path.name


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids, y_true, y_pred, scores = [], [], [], []
    with path.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"ID", "probability", "true_label", "predicted_label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}: {reader.fieldnames}")
        for row in reader:
            try:
                ids.append(row["ID"].strip())
                scores.append(float(row["probability"]))
                y_true.append(int(row["true_label"]))
                y_pred.append(int(row["predicted_label"]))
            except (TypeError, ValueError):
                continue
    if not ids:
        raise ValueError(f"No valid predictions in {path}")
    return (
        np.asarray(ids, dtype=object),
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        np.asarray(scores, dtype=float),
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _fn, _tp = matrix.ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "AUC": float(auc),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "BalancedAccuracy": float((sensitivity + specificity) / 2),
    }


def mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.nanmean([value[key] for value in values])) for key in METRICS}


def std_dict(values: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.nanstd([value[key] for value in values], ddof=0)) for key in METRICS}


def summarize_task(task_dir: Path, filename: str) -> tuple[list[str], dict[str, tuple[dict, dict, dict]]]:
    runs = numbered_dirs(task_dir, "run_")
    if not runs:
        raise ValueError(f"No run_* directories found in {task_dir}")

    run_means = {cohort: [] for cohort in COHORTS}
    run_fold_stds = {cohort: [] for cohort in COHORTS}

    for run_dir in runs:
        folds = numbered_dirs(run_dir, "fold_")
        if not folds:
            raise ValueError(f"No fold_* directories found in {run_dir}")
        fold_metrics = {cohort: [] for cohort in COHORTS}

        for fold_dir in folds:
            result_path = fold_dir / filename
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            ids, y_true, y_pred, scores = load_predictions(result_path)
            masks = {
                "Combined": np.ones(len(ids), dtype=bool),
                "PC-GITA": np.asarray([value.startswith("AVPEPUDE") for value in ids]),
                "NeuroVoz": np.asarray([not value.startswith("AVPEPUDE") for value in ids]),
            }
            for cohort, mask in masks.items():
                if mask.any():
                    fold_metrics[cohort].append(metrics(y_true[mask], y_pred[mask], scores[mask]))

        for cohort in COHORTS:
            if fold_metrics[cohort]:
                run_means[cohort].append(mean_dict(fold_metrics[cohort]))
                run_fold_stds[cohort].append(std_dict(fold_metrics[cohort]))

    summary = {}
    for cohort in COHORTS:
        if run_means[cohort]:
            summary[cohort] = (
                mean_dict(run_means[cohort]),
                std_dict(run_means[cohort]),
                mean_dict(run_fold_stds[cohort]),
            )
    return [path.name for path in runs], summary


def write_summary(
    output_path: Path,
    setting_dir: Path,
    task_dir: Path,
    runs: list[str],
    summary: dict[str, tuple[dict, dict, dict]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(f"Setting: {setting_dir.name}\n")
        stream.write(f"Task: {task_name(task_dir)}\n")
        stream.write(f"Runs: {', '.join(runs)}\n")
        stream.write("Averaging: mean over folds per run, then mean over runs (unweighted).\n")
        stream.write("The probability column is the positive-class probability; saved predicted_label is used for label metrics.\n\n")
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("Cohort", "Metric", "Mean", "RunStd", "MeanFoldStd"))
        for cohort in COHORTS:
            if cohort not in summary:
                continue
            mean, run_std, mean_fold_std = summary[cohort]
            for metric_name in METRICS:
                writer.writerow(
                    (
                        cohort,
                        metric_name,
                        f"{mean[metric_name]:.4f}",
                        f"{run_std[metric_name]:.4f}",
                        f"{mean_fold_std[metric_name]:.4f}",
                    )
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate compact main benchmark results from a setting directory or one task directory."
    )
    parser.add_argument("input_dir", type=Path, help="Setting directory or one folds_tsv_* task directory")
    parser.add_argument(
        "--filename",
        default="Neurovoz_and_PC_GITA_detailed_results.tsv",
        help="Per-fold prediction TSV filename.",
    )
    parser.add_argument("--output-name", default="average_runs_folds.txt")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    setting_dir, tasks = resolve_task_dirs(input_dir)
    if not tasks:
        raise ValueError(f"No folds_tsv_* task directories found in {input_dir}")

    for task_dir in tasks:
        runs, summary = summarize_task(task_dir, args.filename)
        output_path = task_dir / args.output_name
        write_summary(output_path, setting_dir, task_dir, runs, summary)
        print(f"[main] {output_path}")
        print(output_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
