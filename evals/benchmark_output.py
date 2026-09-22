#!/usr/bin/env python3
"""Write external model predictions in the benchmark evaluation format."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DETAILED_FILENAME = "Neurovoz_and_PC_GITA_detailed_results.tsv"
THRESHOLD_FILENAME = "tuned_thresholds.json"
OUTPUT_COLUMNS = (
    "ID",
    "AUDIOFILE",
    "DIAGNOSIS",
    "probability",
    "true_label",
    "predicted_label",
)
TASK_ALIASES = {
    "ddk": "folds_tsv_DDK_ANALYSIS_PATAKA",
    "sentence": "folds_tsv_SENTENCES",
    "sentences": "folds_tsv_SENTENCES",
    "vowel": "folds_tsv_SUSTAINED-VOWELS_onlyA123",
    "vowels": "folds_tsv_SUSTAINED-VOWELS_onlyA123",
}


class BenchmarkResultWriter:
    """Create the task/run/fold outputs consumed by the benchmark evaluators."""

    def __init__(self, setting_dir: str | Path, *, overwrite: bool = False) -> None:
        self.setting_dir = Path(setting_dir).expanduser().resolve()
        self.overwrite = overwrite

    @staticmethod
    def normalize_task(task: str) -> str:
        normalized = task.strip()
        alias = TASK_ALIASES.get(normalized.lower())
        if alias:
            return alias
        if (
            normalized.startswith("folds_tsv_")
            and Path(normalized).name == normalized
            and normalized not in {".", ".."}
        ):
            return normalized
        choices = ", ".join(sorted(TASK_ALIASES))
        raise ValueError(
            f"Unknown task {task!r}; use a folds_tsv_* directory name or one of: {choices}"
        )

    @staticmethod
    def _binary_label(value: Any, field: str, row_number: int) -> int:
        try:
            label = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be 0 or 1 at prediction row {row_number}") from exc
        if label not in (0, 1):
            raise ValueError(f"{field} must be 0 or 1 at prediction row {row_number}")
        return label

    @staticmethod
    def select_threshold(
        validation_records: Iterable[Mapping[str, Any]], *, grid_size: int = 101
    ) -> float:
        """Select the probability threshold with maximum positive-class F1."""
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        labels = []
        probabilities = []
        for row_number, row in enumerate(validation_records, start=1):
            if "probability" not in row or "true_label" not in row:
                raise ValueError(
                    "Validation predictions require probability and true_label "
                    f"at row {row_number}"
                )
            try:
                probability = float(row["probability"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid validation probability at row {row_number}"
                ) from exc
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Validation probability must be within [0, 1] at row {row_number}"
                )
            labels.append(
                BenchmarkResultWriter._binary_label(
                    row["true_label"], "validation true_label", row_number
                )
            )
            probabilities.append(probability)
        if not labels:
            raise ValueError("No validation prediction rows supplied")
        if len(set(labels)) < 2:
            raise ValueError("Validation predictions must contain both classes")

        best_threshold = 0.5
        best_f1 = -1.0
        for index in range(grid_size):
            threshold = index / (grid_size - 1)
            predictions = [int(value >= threshold) for value in probabilities]
            tp = sum(pred == 1 and true == 1 for pred, true in zip(predictions, labels))
            fp = sum(pred == 1 and true == 0 for pred, true in zip(predictions, labels))
            fn = sum(pred == 0 and true == 1 for pred, true in zip(predictions, labels))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        return float(best_threshold)

    @staticmethod
    def select_threshold_from_tsv(
        validation_tsv: str | Path, *, grid_size: int = 101
    ) -> float:
        with Path(validation_tsv).open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream, delimiter="\t"))
        return BenchmarkResultWriter.select_threshold(records, grid_size=grid_size)

    @staticmethod
    def _prepare_rows(
        records: Iterable[Mapping[str, Any]], threshold: float
    ) -> list[dict[str, Any]]:
        output_rows = []
        audiofiles = set()
        for row_number, row in enumerate(records, start=1):
            missing = [
                column
                for column in ("ID", "AUDIOFILE", "DIAGNOSIS", "probability", "true_label")
                if column not in row
            ]
            if missing:
                raise ValueError(
                    f"Missing {', '.join(missing)} at prediction row {row_number}"
                )

            speaker_id = str(row["ID"]).strip()
            audiofile = str(row["AUDIOFILE"]).strip()
            diagnosis = str(row["DIAGNOSIS"]).strip()
            if not speaker_id or not audiofile:
                raise ValueError(f"ID and AUDIOFILE must be non-empty at prediction row {row_number}")
            if audiofile in audiofiles:
                raise ValueError(f"Duplicate AUDIOFILE at prediction row {row_number}: {audiofile}")
            audiofiles.add(audiofile)

            if diagnosis not in ("Healthy", "Parkinson"):
                raise ValueError(
                    f"DIAGNOSIS must be Healthy or Parkinson at prediction row {row_number}"
                )
            try:
                probability = float(row["probability"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid probability at prediction row {row_number}") from exc
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"probability must be finite and within [0, 1] at prediction row {row_number}"
                )

            true_label = BenchmarkResultWriter._binary_label(
                row["true_label"], "true_label", row_number
            )
            expected_true = 1 if diagnosis == "Parkinson" else 0
            if true_label != expected_true:
                raise ValueError(
                    f"DIAGNOSIS and true_label disagree at prediction row {row_number}"
                )

            threshold_prediction = int(probability >= threshold)
            supplied_prediction = row.get("predicted_label", row.get("prediction"))
            if supplied_prediction not in (None, ""):
                predicted_label = BenchmarkResultWriter._binary_label(
                    supplied_prediction, "predicted_label", row_number
                )
                if predicted_label != threshold_prediction:
                    raise ValueError(
                        "predicted_label does not match probability >= threshold "
                        f"at prediction row {row_number}"
                    )
            else:
                predicted_label = threshold_prediction

            output_rows.append(
                {
                    "ID": speaker_id,
                    "AUDIOFILE": audiofile,
                    "DIAGNOSIS": diagnosis,
                    "probability": probability,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                }
            )

        if not output_rows:
            raise ValueError("No prediction rows supplied")
        return output_rows

    def write_fold(
        self,
        *,
        task: str,
        run: int,
        fold: int,
        records: Iterable[Mapping[str, Any]],
        threshold: float,
        threshold_strategy: str = "provided_by_model",
        threshold_grid: int | None = None,
    ) -> tuple[Path, Path]:
        """Validate predictions and write one canonical task/run/fold result."""
        if run < 1 or fold < 1:
            raise ValueError("run and fold must be positive integers")
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and within [0, 1]")

        task_dir = self.normalize_task(task)
        fold_dir = self.setting_dir / task_dir / f"run_{run}" / f"fold_{fold}"
        detailed_path = fold_dir / DETAILED_FILENAME
        threshold_path = fold_dir / THRESHOLD_FILENAME
        existing = [path for path in (detailed_path, threshold_path) if path.exists()]
        if existing and not self.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing benchmark output: "
                + ", ".join(str(path) for path in existing)
            )

        output_rows = self._prepare_rows(records, threshold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        with detailed_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=OUTPUT_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(output_rows)

        threshold_payload: dict[str, Any] = {
            "strategy": threshold_strategy,
            "threshold": threshold,
        }
        if threshold_grid is not None:
            threshold_payload["grid"] = threshold_grid
        threshold_path.write_text(
            json.dumps(
                threshold_payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return detailed_path, threshold_path

    def write_fold_tsv(
        self,
        predictions_tsv: str | Path,
        *,
        task: str,
        run: int,
        fold: int,
        threshold: float,
        threshold_strategy: str = "provided_by_model",
        threshold_grid: int | None = None,
    ) -> tuple[Path, Path]:
        """Read one external prediction TSV and write its canonical fold result."""
        predictions_path = Path(predictions_tsv)
        with predictions_path.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream, delimiter="\t"))
        return self.write_fold(
            task=task,
            run=run,
            fold=fold,
            records=records,
            threshold=threshold,
            threshold_strategy=threshold_strategy,
            threshold_grid=threshold_grid,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions_tsv",
        type=Path,
        help=(
            "Test predictions with ID, AUDIOFILE, DIAGNOSIS, probability, true_label, "
            "and optional predicted_label columns"
        ),
    )
    parser.add_argument("--setting-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        required=True,
        help="Task alias (ddk, vowel, sentence) or a folds_tsv_* directory name",
    )
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    threshold_source = parser.add_mutually_exclusive_group(required=True)
    threshold_source.add_argument(
        "--threshold",
        type=float,
        help="Decision threshold already selected on this fold's validation predictions",
    )
    threshold_source.add_argument(
        "--validation-predictions",
        type=Path,
        help="Validation TSV with probability and true_label columns; selects max positive-class F1",
    )
    parser.add_argument("--threshold-grid", type=int, default=101)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    writer = BenchmarkResultWriter(args.setting_dir, overwrite=args.overwrite)
    if args.validation_predictions is not None:
        threshold = writer.select_threshold_from_tsv(
            args.validation_predictions,
            grid_size=args.threshold_grid,
        )
        threshold_strategy = "max_pos_f1_on_validation"
        threshold_grid = args.threshold_grid
        print(f"Selected validation threshold: {threshold:.6f}")
    else:
        threshold = args.threshold
        threshold_strategy = "provided_by_model"
        threshold_grid = None
    detailed_path, threshold_path = writer.write_fold_tsv(
        args.predictions_tsv,
        task=args.task,
        run=args.run,
        fold=args.fold,
        threshold=threshold,
        threshold_strategy=threshold_strategy,
        threshold_grid=threshold_grid,
    )
    print(f"Wrote benchmark predictions: {detailed_path}")
    print(f"Wrote validation threshold: {threshold_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
