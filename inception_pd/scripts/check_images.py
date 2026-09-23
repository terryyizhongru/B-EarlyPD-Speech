#!/usr/bin/env python3
"""Check the spectrogram JPGs required by one benchmark training setting."""

import argparse
import csv
import json
from pathlib import Path
from PIL import Image

TASKS = (
    "folds_tsv_DDK_ANALYSIS_PATAKA",
    "folds_tsv_SUSTAINED-VOWELS_onlyA123",
    "folds_tsv_SENTENCES",
)
SETTINGS = {
    "all": ("train.tsv", "val_early6PD6HC.tsv", "test_early6PD6HC.tsv"),
    "all_subset": ("train_allPDsubset.tsv", "val_early6PD6HC.tsv", "test_early6PD6HC.tsv"),
    "early": ("train_earlybalance.tsv", "val_early6PD6HC.tsv", "test_early6PD6HC.tsv"),
    "early_private": ("train_early_persp.tsv", "val_early6PD6HC.tsv", "test_early6PD6HC.tsv"),
    "all_stage": ("train.tsv", "val_all6PD6HC.tsv", "test_all6PD6HC.tsv"),
}


def image_path(wav: str) -> Path:
    marker = "/audios_fortrain/"
    if marker not in wav or not wav.lower().endswith(".wav"):
        raise ValueError(f"Expected a WAV path under audios_fortrain/: {wav}")
    return Path(wav.replace(marker, "/audios_fortrain_jpg/").rsplit(".", 1)[0] + ".jpg")


def split_list(root: Path, task: str, fold: int, filename: str) -> Path:
    base = root / task / f"fold_{fold}"
    if filename.startswith("test_"):
        return base / filename
    for dirname in ("train_and_val", "sub_splits"):
        candidate = base / dirname / filename
        if candidate.exists():
            return candidate
    return base / "train_and_val" / filename


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--setting", choices=SETTINGS, required=True)
    parser.add_argument("--train-splits-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    train_root = args.train_splits_dir or args.splits_dir
    if args.setting == "early_private" and args.train_splits_dir is None:
        parser.error("early_private requires --train-splits-dir")
    train_name, val_name, test_name = SETTINGS[args.setting]
    tasks = TASKS[:2] if args.setting == "early_private" else TASKS
    problems = []
    images = set()
    checked_lists = 0
    for task in tasks:
        for fold in range(1, 6):
            for root, name in ((train_root, train_name), (args.splits_dir, val_name), (args.splits_dir, test_name)):
                path = split_list(root, task, fold, name)
                if not path.is_file():
                    problems.append(f"Missing TSV: {path}")
                    continue
                checked_lists += 1
                with path.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if not reader.fieldnames or "AUDIOFILE" not in reader.fieldnames:
                        problems.append(f"Missing AUDIOFILE column: {path}")
                        continue
                    for row in reader:
                        try:
                            jpg = image_path(row["AUDIOFILE"])
                        except (TypeError, ValueError) as exc:
                            problems.append(str(exc))
                            continue
                        images.add(jpg)
    for jpg in sorted(images):
        if not jpg.is_file():
            problems.append(f"Missing JPG: {jpg}")
            continue
        try:
            with Image.open(jpg) as image:
                image.verify()
        except Exception as exc:
            problems.append(f"Unreadable JPG: {jpg}: {exc}")
    report = {
        "setting": args.setting,
        "checked_lists": checked_lists,
        "unique_images": len(images),
        "errors": len(problems),
        "examples": problems[:50],
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
