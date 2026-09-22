#!/usr/bin/env python3
"""Aggregate gender-level predictions from a setting or one task directory."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

METRICS = ("F1", "AUC")


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


def load_gender_map(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"ID", "Gender"}.issubset(reader.fieldnames):
            raise ValueError(f"Metadata must contain ID and Gender: {path}")
        for row in reader:
            sample_id = (row.get("ID") or "").strip()
            gender = (row.get("Gender") or "").strip()
            if sample_id and gender:
                mapping[sample_id] = gender
    return mapping


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
    return (
        np.asarray(ids, dtype=object),
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        np.asarray(scores, dtype=float),
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "AUC": float(auc),
    }


def mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.nanmean([value[key] for value in values])) for key in METRICS}


def std_dict(values: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.nanstd([value[key] for value in values], ddof=0)) for key in METRICS}


def aggregate_task(
    task_dir: Path,
    gender_map: dict[str, str],
    filename: str,
    genders: list[str],
) -> tuple[list[str], dict[str, tuple[dict, dict, dict]]]:
    runs = numbered_dirs(task_dir, "run_")
    if not runs:
        raise ValueError(f"No run_* directories found in {task_dir}")
    run_means = {gender: [] for gender in genders}
    run_fold_stds = {gender: [] for gender in genders}

    for run_dir in runs:
        folds = numbered_dirs(run_dir, "fold_")
        if not folds:
            raise ValueError(f"No fold_* directories found in {run_dir}")
        fold_metrics = {gender: [] for gender in genders}
        for fold_dir in folds:
            result_path = fold_dir / filename
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            ids, y_true, y_pred, scores = load_predictions(result_path)
            sample_genders = np.asarray([gender_map.get(value, "") for value in ids])
            for gender in genders:
                mask = sample_genders == gender
                if mask.any():
                    fold_metrics[gender].append(metrics(y_true[mask], y_pred[mask], scores[mask]))
        for gender in genders:
            if fold_metrics[gender]:
                run_means[gender].append(mean_dict(fold_metrics[gender]))
                run_fold_stds[gender].append(std_dict(fold_metrics[gender]))

    summary = {}
    for gender in genders:
        if run_means[gender]:
            summary[gender] = (
                mean_dict(run_means[gender]),
                std_dict(run_means[gender]),
                mean_dict(run_fold_stds[gender]),
            )
    return [path.name for path in runs], summary


def write_summary(
    output_path: Path,
    setting_dir: Path,
    task_dir: Path,
    meta_csv: Path,
    runs: list[str],
    genders: list[str],
    summary: dict[str, tuple[dict, dict, dict]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(f"Setting: {setting_dir.name}\n")
        stream.write(f"Task: {task_dir.name}\n")
        stream.write(f"Runs: {', '.join(runs)}\n")
        stream.write(f"Metadata: {meta_csv}\n")
        stream.write("Averaging: mean over folds per run, then mean over runs (unweighted).\n\n")
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("Gender", "Mean_F1", "RunStd_F1", "MeanFoldStd_F1", "Mean_AUC", "RunStd_AUC", "MeanFoldStd_AUC"))
        for gender in genders:
            if gender not in summary:
                writer.writerow((gender, "NA", "NA", "NA", "NA", "NA", "NA"))
                continue
            mean, run_std, mean_fold_std = summary[gender]
            writer.writerow(
                (
                    gender,
                    f"{mean['F1']:.4f}",
                    f"{run_std['F1']:.4f}",
                    f"{mean_fold_std['F1']:.4f}",
                    f"{mean['AUC']:.4f}",
                    f"{run_std['AUC']:.4f}",
                    f"{mean_fold_std['AUC']:.4f}",
                )
            )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Generate gender-level results from a setting directory or one task directory."
    )
    parser.add_argument("input_dir", type=Path, help="Setting directory or one folds_tsv_* task directory")
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=repo_root / "evals/metadata/speaker_gender.csv",
        help="CSV containing ID and Gender columns.",
    )
    parser.add_argument("--filename", default="Neurovoz_and_PC_GITA_detailed_results.tsv")
    parser.add_argument("--output-name", default="average_gender_runs_folds.txt")
    parser.add_argument("--genders", nargs="+", default=["M", "F"])
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    meta_csv = args.meta_csv.resolve()
    try:
        metadata_for_report = meta_csv.relative_to(repo_root)
    except ValueError:
        metadata_for_report = meta_csv
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    gender_map = load_gender_map(meta_csv)
    setting_dir, tasks = resolve_task_dirs(input_dir)
    if not tasks:
        raise ValueError(f"No folds_tsv_* task directories found in {input_dir}")

    for task_dir in tasks:
        runs, summary = aggregate_task(task_dir, gender_map, args.filename, args.genders)
        output_path = task_dir / args.output_name
        write_summary(output_path, setting_dir, task_dir, metadata_for_report, runs, args.genders, summary)
        print(f"[gender] {output_path}")
        print(output_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
