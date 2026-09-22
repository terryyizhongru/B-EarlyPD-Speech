#!/usr/bin/env python3
"""Add private DDK and vowel recordings to every EarlyPD training fold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLUMNS = ("ID", "AUDIOFILE", "DIAGNOSIS")
TASK_INPUTS = {
    "folds_tsv_DDK_ANALYSIS_PATAKA": "private_ddk",
    "folds_tsv_SUSTAINED-VOWELS_onlyA123": "private_vowel",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(COLUMNS)}")
        rows = [{column: (row.get(column) or "").strip() for column in COLUMNS} for row in reader]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def train_dir(root: Path, task: str, fold: int) -> Path:
    base = root / task / f"fold_{fold}"
    for name in ("train_and_val", "sub_splits"):
        candidate = base / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing train directory under {base}")


def public_inventory(root: Path) -> tuple[set[str], set[str]]:
    speaker_ids: set[str] = set()
    audiofiles: set[str] = set()
    files = sorted(root.rglob("*.tsv"))
    if not files:
        raise ValueError(f"No TSV files found under {root}")
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = set(reader.fieldnames or ())
            if not {"ID", "AUDIOFILE"}.issubset(fields):
                continue
            for row in reader:
                speaker_id = (row.get("ID") or "").strip()
                audiofile = (row.get("AUDIOFILE") or "").strip()
                if speaker_id:
                    speaker_ids.add(speaker_id)
                if audiofile:
                    audiofiles.add(audiofile)
    return speaker_ids, audiofiles


def validate_private_rows(
    task_rows: dict[str, list[dict[str, str]]],
    public_ids: set[str],
    public_audiofiles: set[str],
) -> None:
    labels: dict[str, str] = {}
    seen_audiofiles: set[str] = set()
    errors: list[str] = []

    for task, rows in task_rows.items():
        for index, row in enumerate(rows, start=2):
            context = f"{task}, line {index}"
            speaker_id = row["ID"]
            audiofile = row["AUDIOFILE"]
            diagnosis = row["DIAGNOSIS"]
            if not speaker_id or not audiofile:
                errors.append(f"{context}: ID and AUDIOFILE must be non-empty")
                continue
            if diagnosis not in ("Healthy", "Parkinson"):
                errors.append(f"{context}: DIAGNOSIS must be Healthy or Parkinson")
            if "audios_fortrain" not in Path(audiofile).parts:
                errors.append(f"{context}: AUDIOFILE must be under audios_fortrain")
            if speaker_id in public_ids:
                errors.append(f"{context}: private ID overlaps a public speaker: {speaker_id}")
            if audiofile in public_audiofiles:
                errors.append(f"{context}: private AUDIOFILE overlaps a public recording: {audiofile}")
            if audiofile in seen_audiofiles:
                errors.append(f"{context}: duplicate private AUDIOFILE: {audiofile}")
            seen_audiofiles.add(audiofile)
            previous = labels.setdefault(speaker_id, diagnosis)
            if previous != diagnosis:
                errors.append(
                    f"{context}: conflicting labels for {speaker_id}: {previous} vs {diagnosis}"
                )

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:50])
        suffix = "" if len(errors) <= 50 else f"\n- ... {len(errors) - 50} more"
        raise ValueError(f"Invalid private training data:\n{preview}{suffix}")


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--private-ddk", type=Path, required=True)
    parser.add_argument("--private-vowel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    splits_root = args.splits_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists():
        parser.error(f"--output-dir must not already exist: {output_root}")
    if output_root == splits_root or splits_root in output_root.parents:
        parser.error("--output-dir must be outside --splits-dir")

    task_rows = {
        task: read_tsv(getattr(args, argument).expanduser().resolve())
        for task, argument in TASK_INPUTS.items()
    }
    public_ids, public_audiofiles = public_inventory(splits_root)
    validate_private_rows(task_rows, public_ids, public_audiofiles)

    generated = []
    for task, private_rows in task_rows.items():
        for fold in range(1, 6):
            source = train_dir(splits_root, task, fold) / "train_earlybalance.tsv"
            public_rows = read_tsv(source)
            target = (
                output_root
                / task
                / f"fold_{fold}"
                / "train_and_val"
                / "train_early_persp.tsv"
            )
            write_tsv(target, public_rows + private_rows)
            generated.append(
                {
                    "task": task,
                    "fold": fold,
                    "source_rows": len(public_rows),
                    "private_rows": len(private_rows),
                    "output_rows": len(public_rows) + len(private_rows),
                    "output": str(target),
                }
            )

    report = {
        "splits_dir": str(splits_root),
        "output_dir": str(output_root),
        "private_speakers": len(
            {row["ID"] for rows in task_rows.values() for row in rows}
        ),
        "private_recordings": sum(len(rows) for rows in task_rows.values()),
        "generated_files": generated,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "private_split_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
