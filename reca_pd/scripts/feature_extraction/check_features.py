#!/usr/bin/env python3
"""Check the audio and feature files required by one RECA-PD experiment."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path


TASKS = (
    "folds_tsv_DDK_ANALYSIS_PATAKA",
    "folds_tsv_SUSTAINED-VOWELS_onlyA123",
    "folds_tsv_SENTENCES",
)

FEATURE_DIRS = {
    "wav2vec_layer07": "speech_features/wav2vec/layer07",
    "articulation": "speech_features/disvoice/articulation",
    "glottal": "speech_features/disvoice/glottal",
    "phonation": "speech_features/disvoice/phonation",
    "prosody": "speech_features/disvoice/prosody",
}

REGIMES = {
    "all": ("train.tsv", "val_early6PD6HC.tsv", "test_early6PD6HC.tsv"),
    "all_subset": ("train_allPDsubset.tsv", "val_early6PD6HC.tsv", "test_early6PD6HC.tsv"),
    "early": ("train_earlybalance.tsv", "val_early6PD6HC.tsv", "test_early6PD6HC.tsv"),
    "early_private": ("train_early_persp.tsv", "val_early6PD6HC.tsv", "test_early6PD6HC.tsv"),
    "all_stage": ("train.tsv", "val_all6PD6HC.tsv", "test_all6PD6HC.tsv"),
}


def split_dir(root: Path, task: str, fold: int) -> Path:
    """Return the train/validation directory used by the launch scripts."""
    base = root / task / f"fold_{fold}"
    for name in ("train_and_val", "sub_splits"):
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return base / "train_and_val"


def read_audio_paths(path: Path) -> tuple[list[str], str | None]:
    if not path.is_file():
        return [], f"Missing split: {path}"
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames or "AUDIOFILE" not in reader.fieldnames:
                return [], f"Missing AUDIOFILE column: {path}"
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        return [], f"Could not read split {path}: {exc}"
    if not rows:
        return [], f"Empty split: {path}"
    paths = [row.get("AUDIOFILE", "").strip() for row in rows]
    if any(not audio for audio in paths):
        return [], f"Empty AUDIOFILE value: {path}"
    return paths, None


def feature_path(audio_path: str, feature_dir: str) -> Path | None:
    marker = "audios_fortrain"
    if audio_path.count(marker) != 1:
        return None
    return Path(audio_path.replace(marker, feature_dir)).with_suffix(".npz")


def check_wav(path: Path) -> str | None:
    try:
        with wave.open(str(path), "rb") as wav:
            actual = (
                wav.getnchannels(),
                wav.getframerate(),
                wav.getsampwidth(),
                wav.getcomptype(),
            )
            if actual != (1, 16000, 2, "NONE") or wav.getnframes() == 0:
                return "Expected non-empty mono 16 kHz 16-bit PCM WAV"
    except (OSError, EOFError, wave.Error) as exc:
        return str(exc)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument(
        "--train-splits-dir",
        type=Path,
        help="Optional separate training split root, used for EarlyPD + private data",
    )
    parser.add_argument("--regime", choices=sorted(REGIMES), default="all")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    splits_root = args.splits_dir.resolve()
    train_root = (args.train_splits_dir or args.splits_dir).resolve()
    train_name, val_name, test_name = REGIMES[args.regime]
    tasks = TASKS[:-1] if args.regime == "early_private" else TASKS

    split_errors: list[str] = []
    split_files: list[dict[str, object]] = []
    audio_paths: set[str] = set()

    for task in tasks:
        for fold in range(1, 6):
            train_path = split_dir(train_root, task, fold) / train_name
            val_path = split_dir(splits_root, task, fold) / val_name
            test_path = splits_root / task / f"fold_{fold}" / test_name
            for role, path in (
                ("train", train_path),
                ("validation", val_path),
                ("test", test_path),
            ):
                paths, error = read_audio_paths(path)
                split_files.append(
                    {
                        "task": task,
                        "fold": fold,
                        "role": role,
                        "path": str(path),
                        "rows": len(paths),
                    }
                )
                if error:
                    split_errors.append(error)
                else:
                    audio_paths.update(paths)

    audio_errors: list[dict[str, str]] = []
    feature_errors: list[dict[str, str]] = []
    feature_counts = {name: 0 for name in FEATURE_DIRS}

    for audio_name in sorted(audio_paths):
        audio = Path(audio_name)
        audio_error = check_wav(audio)
        if audio_error:
            audio_errors.append({"path": audio_name, "error": audio_error})

        for feature_name, directory in FEATURE_DIRS.items():
            path = feature_path(audio_name, directory)
            if path is None:
                feature_errors.append(
                    {
                        "audio": audio_name,
                        "feature": feature_name,
                        "path": "",
                        "error": "AUDIOFILE must contain audios_fortrain exactly once",
                    }
                )
            elif not path.is_file():
                feature_errors.append(
                    {
                        "audio": audio_name,
                        "feature": feature_name,
                        "path": str(path),
                        "error": "Missing feature file",
                    }
                )
            elif path.stat().st_size == 0:
                feature_errors.append(
                    {
                        "audio": audio_name,
                        "feature": feature_name,
                        "path": str(path),
                        "error": "Empty feature file",
                    }
                )
            else:
                feature_counts[feature_name] += 1

    report = {
        "regime": args.regime,
        "splits_dir": str(splits_root),
        "train_splits_dir": str(train_root),
        "split_files": split_files,
        "unique_audio_paths": len(audio_paths),
        "feature_counts": feature_counts,
        "split_errors": split_errors,
        "audio_errors": audio_errors,
        "feature_errors": feature_errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    summary = {
        "regime": args.regime,
        "split_errors": len(split_errors),
        "audio_errors": len(audio_errors),
        "feature_errors": len(feature_errors),
        "unique_audio_paths": len(audio_paths),
        "report": str(args.report),
    }
    print(json.dumps(summary))
    raise SystemExit(1 if split_errors or audio_errors or feature_errors else 0)


if __name__ == "__main__":
    main()
