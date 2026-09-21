#!/usr/bin/env python3
"""Benchmark audio conversion: SoX peak normalization, mono, 16 kHz, 16-bit PCM."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav-dir', type=Path, required=True, help='Flat directory of renamed WAV files')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; source files are preserved')
    parser.add_argument('--norm-db', type=float, default=-3.0, help='Peak normalization level used by existing preprocessing')
    args = parser.parse_args()
    if not shutil.which('sox'):
        parser.error('SoX is required')
    if not args.wav_dir.is_dir() or args.output_dir.exists():
        parser.error('Input must exist and output must be a new directory')
    wavs = sorted(p for p in args.wav_dir.iterdir() if p.suffix.lower() == '.wav' and p.is_file())
    if not wavs:
        parser.error('No WAV files in input directory')
    args.output_dir.mkdir(parents=True)
    for wav in wavs:
        target = args.output_dir / wav.name
        subprocess.run(['sox', f'--norm={args.norm_db}', str(wav), '-b', '16', '-c', '1', '-r', '16000', '-e', 'signed-integer', str(target)], check=True)
    print(json.dumps({'processed': len(wavs), 'output': str(args.output_dir.resolve()), 'norm_db': args.norm_db}))


if __name__ == '__main__':
    main()
