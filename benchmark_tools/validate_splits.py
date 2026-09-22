#!/usr/bin/env python3
"""Audit paper single-task splits without changing them. Python standard library only."""
import argparse
import collections
import csv
import json
import wave
from pathlib import Path

TASKS = ['folds_tsv_DDK_ANALYSIS_PATAKA', 'folds_tsv_SUSTAINED-VOWELS_onlyA123', 'folds_tsv_SENTENCES']
TRAIN = ['train.tsv', 'train_earlybalance.tsv', 'train_allPDsubset.tsv']


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t' if path.suffix == '.tsv' else ','))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-dir', type=Path, required=True)
    parser.add_argument('--check-audio', action='store_true', help='Check existence and mono/16 kHz/16-bit PCM WAV headers')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.splits_dir
    errors, warnings, summaries, paths = [], [], [], set()
    for task in TASKS:
        tests = set()
        for fold in range(1, 6):
            base = root / task / f'fold_{fold}'
            loaded = {}
            names = ['test_early6PD6HC.tsv', 'test_all6PD6HC.tsv'] + ['train_and_val/' + n for n in TRAIN + ['val_early6PD6HC.tsv', 'val_all6PD6HC.tsv']]
            for name in names:
                f = base / name
                if not f.is_file():
                    errors.append(f'Missing split: {f}')
                    continue
                rows = read(f)
                if not rows or not {'ID', 'AUDIOFILE', 'DIAGNOSIS'} <= rows[0].keys():
                    errors.append(f'Empty or invalid schema: {f}')
                    continue
                loaded[name] = rows
                tuples = [(r['ID'], r['AUDIOFILE'], r['DIAGNOSIS']) for r in rows]
                if len(set(tuples)) != len(tuples):
                    errors.append(f'Duplicate rows: {f}')
                if any(r['DIAGNOSIS'] not in ('Healthy', 'Parkinson') or not r['ID'] for r in rows):
                    errors.append(f'Invalid ID/diagnosis: {f}')
                labels = collections.defaultdict(set)
                for r in rows:
                    labels[r['ID']].add(r['DIAGNOSIS'])
                    paths.add(r['AUDIOFILE'])
                if any(len(v) != 1 for v in labels.values()):
                    errors.append(f'Conflicting speaker labels: {f}')
                summaries.append({'task': task, 'fold': fold, 'split': name, 'rows': len(rows), 'speakers': len(labels), 'speaker_classes': dict(collections.Counter(next(iter(v)) for v in labels.values()))})
            ids = lambda name: {r['ID'] for r in loaded.get(name, [])}
            val, test = ids('train_and_val/val_early6PD6HC.tsv'), ids('test_early6PD6HC.tsv')
            for name in TRAIN:
                overlap = ids('train_and_val/' + name) & (val | test)
                if overlap:
                    errors.append(f'{task}/fold_{fold}/{name}: train vs early val/test overlap {sorted(overlap)}')
            if val & test:
                errors.append(f'{task}/fold_{fold}: early val-test overlap')
            if tests & test:
                errors.append(f'{task}/fold_{fold}: repeated early test speaker across folds')
            tests |= test
            for kind, actual in [('test', test), ('val', val)]:
                relative = f'test_early6PD6HC.csv' if kind == 'test' else 'train_and_val/val_early6PD6HC.csv'
                expected = {r['ID'] for r in read(root / 'folds_csv' / f'fold_{fold}' / relative)}
                if actual - expected:
                    errors.append(f'{task}/fold_{fold}/{kind}: unexpected speakers {sorted(actual-expected)}')
                if expected - actual:
                    warnings.append(f'{task}/fold_{fold}/{kind}: speaker CSV IDs absent in task TSV {sorted(expected-actual)}; cause not yet established')
            # Report canonical cohort differences; do not silently repair them.
            baseline = {r['ID'] for r in loaded.get('train_and_val/train.tsv', []) if r['DIAGNOSIS'] == 'Healthy'}
            for name in TRAIN[1:]:
                hc = {r['ID'] for r in loaded.get('train_and_val/' + name, []) if r['DIAGNOSIS'] == 'Healthy'}
                if hc != baseline:
                    warnings.append(f'{task}/fold_{fold}/{name}: HC cohort differs from AllPD (only_this={sorted(hc-baseline)}, only_AllPD={sorted(baseline-hc)})')
    audio_errors = []
    if args.check_audio:
        for name in sorted(paths):
            try:
                with wave.open(name, 'rb') as wav:
                    if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, 'NONE') or wav.getnframes() == 0:
                        audio_errors.append({'path': name, 'error': 'Expected non-empty mono 16 kHz 16-bit PCM WAV'})
            except (OSError, wave.Error, EOFError) as exc:
                audio_errors.append({'path': name, 'error': str(exc)})
    report = {'scope': 'Three paper tasks; all-stage files inventoried, leakage checks cover EarlyPD evaluation only', 'errors': errors, 'warnings': warnings, 'splits': summaries, 'unique_audio_paths': len(paths), 'audio_checked': args.check_audio, 'audio_errors': audio_errors}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'split_errors': len(errors), 'warnings': len(warnings), 'audio_errors': len(audio_errors), 'unique_audio_paths': len(paths), 'report': str(args.report)}))
    raise SystemExit(1 if errors or audio_errors else 0)


if __name__ == '__main__':
    main()
