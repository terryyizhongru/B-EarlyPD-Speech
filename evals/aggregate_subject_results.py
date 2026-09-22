#!/usr/bin/env python3
"""Aggregate subject-level predictions from a setting or one task directory."""

import argparse
import csv
import hashlib
import json
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


def load_threshold(fold_dir: Path, candidates: list[str], key: str) -> float:
    errors = []
    for name in candidates:
        path = fold_dir / name
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "threshold" in obj:
                return float(obj["threshold"])
            thresholds = obj.get("thresholds") if isinstance(obj, dict) else None
            if isinstance(thresholds, dict):
                if key in thresholds:
                    return float(thresholds[key])
                if len(thresholds) == 1:
                    return float(next(iter(thresholds.values())))
            raise ValueError(f"Unsupported threshold schema: {path}")
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise ValueError("; ".join(errors))
    raise FileNotFoundError(f"No threshold JSON found in {fold_dir}")


def load_rows(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids, y_true, scores = [], [], []
    with path.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"ID", "probability", "true_label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}: {reader.fieldnames}")
        for row in reader:
            try:
                ids.append(row["ID"].strip())
                scores.append(float(row["probability"]))
                y_true.append(int(row["true_label"]))
            except (TypeError, ValueError):
                continue
    if not ids:
        raise ValueError(f"No valid predictions in {path}")
    return ids, np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "AUC": float(auc),
    }


def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def subject_metrics(
    ids: list[str],
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sample_k: int,
    sample_seed: int,
    sample_salt: str,
) -> dict[str, float]:
    by_id_scores: dict[str, list[float]] = {}
    by_id_true: dict[str, list[int]] = {}
    for sample_id, true, score in zip(ids, y_true.tolist(), scores.tolist()):
        by_id_scores.setdefault(sample_id, []).append(float(score))
        by_id_true.setdefault(sample_id, []).append(int(true))

    subject_true, subject_scores = [], []
    for sample_id in sorted(by_id_scores):
        available = by_id_scores[sample_id]
        labels = np.asarray(by_id_true[sample_id], dtype=int)
        values, counts = np.unique(labels, return_counts=True)
        subject_true.append(int(values[int(np.argmax(counts))]))

        if sample_k <= 0 or len(available) <= sample_k:
            selected = np.asarray(available, dtype=float)
        else:
            seed = (sample_seed ^ stable_seed(f"{sample_salt}|{sample_id}")) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(available), size=sample_k, replace=False)
            selected = np.asarray([available[index] for index in indices], dtype=float)
        subject_scores.append(float(np.mean(selected)))

    true_array = np.asarray(subject_true, dtype=int)
    score_array = np.asarray(subject_scores, dtype=float)
    pred_array = (score_array >= threshold).astype(int)
    return binary_metrics(true_array, pred_array, score_array)


def mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.nanmean([value[key] for value in values])) for key in METRICS}


def aggregate_task(
    setting_dir: Path,
    task_dir: Path,
    filename: str,
    threshold_names: list[str],
    threshold_key: str,
    sample_ks: list[int],
    sample_seed: int,
    sample_salt_root: Path,
) -> tuple[list[str], dict[str, float], dict[int, dict[str, float]]]:
    runs = numbered_dirs(task_dir, "run_")
    if not runs:
        raise ValueError(f"No run_* directories found in {task_dir}")

    utterance_run_means = []
    subject_run_means = {sample_k: [] for sample_k in sample_ks}

    for run_dir in runs:
        folds = numbered_dirs(run_dir, "fold_")
        if not folds:
            raise ValueError(f"No fold_* directories found in {run_dir}")
        utterance_folds = []
        subject_folds = {sample_k: [] for sample_k in sample_ks}

        for fold_dir in folds:
            result_path = fold_dir / filename
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            threshold = load_threshold(fold_dir, threshold_names, threshold_key)
            ids, y_true, scores = load_rows(result_path)
            y_pred = (scores >= threshold).astype(int)
            utterance_folds.append(binary_metrics(y_true, y_pred, scores))

            historical_fold = (
                sample_salt_root
                / setting_dir.name
                / task_dir.name
                / run_dir.name
                / fold_dir.name
            )
            for sample_k in sample_ks:
                subject_folds[sample_k].append(
                    subject_metrics(
                        ids,
                        y_true,
                        scores,
                        threshold,
                        sample_k,
                        sample_seed,
                        str(historical_fold),
                    )
                )

        utterance_run_means.append(mean_metrics(utterance_folds))
        for sample_k in sample_ks:
            subject_run_means[sample_k].append(mean_metrics(subject_folds[sample_k]))

    utterance_mean = mean_metrics(utterance_run_means)
    subject_means = {
        sample_k: mean_metrics(subject_run_means[sample_k]) for sample_k in sample_ks
    }
    return [path.name for path in runs], utterance_mean, subject_means


def write_result(
    output_path: Path,
    task_dir: Path,
    runs: list[str],
    sample_k: int,
    sample_seed: int,
    mean: dict[str, float],
    utterance_mean: dict[str, float],
) -> None:
    delta_f1 = mean["F1"] - utterance_mean["F1"]
    delta_auc = mean["AUC"] - utterance_mean["AUC"]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("root", "runs", "sample_k", "sample_seed", "Mean_F1", "Mean_AUC", "Delta_F1", "Delta_AUC"))
        writer.writerow(
            (
                task_dir.name,
                ",".join(runs),
                sample_k,
                sample_seed,
                f"{mean['F1']:.4f}",
                f"{mean['AUC']:.4f}",
                f"{delta_f1:.4f}",
                f"{delta_auc:.4f}",
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate subject-level results from a setting directory or one task directory."
    )
    parser.add_argument("input_dir", type=Path, help="Setting directory or one folds_tsv_* task directory")
    parser.add_argument("--filename", default="Neurovoz_and_PC_GITA_detailed_results.tsv")
    parser.add_argument(
        "--threshold-json",
        default="tuned_thresholds.json,tuned_threshold.json",
        help="Comma-separated candidate filenames.",
    )
    parser.add_argument("--threshold-key", default="Neurovoz_and_PC_GITA")
    parser.add_argument("--sentence-k", nargs="+", type=int, default=[3, 10])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-salt-root",
        type=Path,
        default=Path("/data/storage1t/gits/interpretable-pd/exps"),
        help="Historical logical experiments root used as the deterministic sampling namespace.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    setting_dir, tasks = resolve_task_dirs(input_dir)
    if not tasks:
        raise ValueError(f"No folds_tsv_* task directories found in {input_dir}")
    sentence_tasks = [task for task in tasks if "SENTENCES" in task.name.upper()]
    vowel_tasks = [task for task in tasks if "SUSTAINED-VOWELS" in task.name.upper()]
    if not sentence_tasks and not vowel_tasks:
        print(f"[subject] no Sentence or Sustained-Vowel task in {input_dir}; skipped")
        return 0
    threshold_names = [name.strip() for name in args.threshold_json.split(",") if name.strip()]

    for task_dir in sentence_tasks:
        runs, utterance_mean, subject_means = aggregate_task(
            setting_dir,
            task_dir,
            args.filename,
            threshold_names,
            args.threshold_key,
            args.sentence_k,
            args.sample_seed,
            args.sample_salt_root,
        )
        for sample_k in args.sentence_k:
            output_path = task_dir / f"average_subjectlevel_sentences_k{sample_k}.tsv"
            write_result(
                output_path,
                task_dir,
                runs,
                sample_k,
                args.sample_seed,
                subject_means[sample_k],
                utterance_mean,
            )
            print(f"[subject:sentence] {output_path}")
            print(output_path.read_text(encoding="utf-8"), end="")

    for task_dir in vowel_tasks:
        runs, utterance_mean, subject_means = aggregate_task(
            setting_dir,
            task_dir,
            args.filename,
            threshold_names,
            args.threshold_key,
            [0],
            0,
            args.sample_salt_root,
        )
        output_path = task_dir / "average_subjectlevel_vowels.tsv"
        write_result(
            output_path,
            task_dir,
            runs,
            0,
            0,
            subject_means[0],
            utterance_mean,
        )
        print(f"[subject:vowel] {output_path}")
        print(output_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
