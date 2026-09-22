#!/usr/bin/env python3
"""Materialize canonical splits with local audio paths; never change membership."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'benchmark_splits')
    p.add_argument('--data-root', type=Path, required=True, help='Parent of pcgita_data/ and neurovoz_data/')
    p.add_argument('--output-dir', type=Path, required=True, help='New, non-existing directory')
    a = p.parse_args()
    source, output, data = a.splits_dir.resolve(), a.output_dir.resolve(), a.data_root.resolve()
    if output.exists() or output == source or source in output.parents:
        p.error('Output must be a new directory outside the canonical splits tree')
    files = sorted(source.rglob('*.csv')) + sorted(source.rglob('*.tsv'))
    if not files:
        p.error('No split files found')
    prepared = []
    for f in files:
        if f.suffix == '.csv':
            prepared.append((f.relative_to(source), f.read_bytes()))
            continue
        with f.open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            fields, rows = reader.fieldnames, list(reader)
        if not fields or 'AUDIOFILE' not in fields:
            raise ValueError(f'Missing AUDIOFILE: {f}')
        for row in rows:
            parts = Path(row['AUDIOFILE']).parts
            indices = [i for i, part in enumerate(parts) if part in ('pcgita_data', 'neurovoz_data')]
            if len(indices) != 1 or '..' in parts:
                raise ValueError(f'Unrecognized audio path: {row["AUDIOFILE"]}')
            row['AUDIOFILE'] = str(data.joinpath(*parts[indices[0]:]))
        prepared.append((f.relative_to(source), (fields, rows)))
    output.mkdir(parents=True)
    for relative, content in prepared:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            fields, rows = content
            with target.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({'files': len(files), 'source': str(source), 'data_root': str(data), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
