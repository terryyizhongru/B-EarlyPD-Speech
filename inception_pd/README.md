# Inception-PD benchmark reproduction

This directory adapts the InceptionV3 speech-spectrogram method from `/data/storage2/gits/CNN-PD-Voice` at revision `bbc514630d0425c8e9d8b2569d87828dd633e69c`. The model code retains its original attribution and [Apache 2.0 license](LICENSE).

## 1. Prepare audio and spectrograms

First follow the [main benchmark guide](../README.md#2-prepare-audio) to prepare mono 16 kHz WAVs and [generate local filelists](../README.md#3-generate-and-validate-local-filelists). Install the model dependencies in a Python 3.10 environment:

```bash
pip install -r inception_pd/requirements.txt
```

The model reads a JPG for every `AUDIOFILE` WAV by replacing the directory component `audios_fortrain/` with `audios_fortrain_jpg/` and `.wav` with `.jpg`. Generate the images for both datasets, replacing `PD_DATA` with the prepared-data directory used in the main guide:

```bash
PD_DATA="/path/to/earlypd-prepared/NeuroVoz_PCGITA"

python inception_pd/audio2image.py \
  --in_dir "$PD_DATA/neurovoz_data/audios_fortrain" \
  --out_dir "$PD_DATA/neurovoz_data/audios_fortrain_jpg" \
  --type mel

python inception_pd/audio2image.py \
  --in_dir "$PD_DATA/pcgita_data/audios_fortrain" \
  --out_dir "$PD_DATA/pcgita_data/audios_fortrain_jpg" \
  --type mel
```

The script uses 16 kHz audio, 10-second padding or truncation, and 600 × 600 JPG output by default. Its default mel mode reproduced an existing archived JPG exactly in a pixel comparison. Check all images required for the selected setting before training:

```bash
python inception_pd/scripts/check_images.py \
  --splits-dir "/path/to/runtime_splits" \
  --setting all \
  --report "/path/to/inception_image_audit.json"
```

The check exits with an error for missing or unreadable images.

## 2. Select an experiment

All launchers use seeds 42–46 and five folds. Public settings cover DDK, sustained vowel /a/, and Sentence; the Private track covers DDK and vowel only.

| Launcher | Training TSV | Validation / test |
|---|---|---|
| `run_5seeds_all.sh` | `train.tsv` | EarlyPD |
| `run_5seeds_allsubset.sh` | `train_allPDsubset.tsv` | EarlyPD |
| `run_5seeds_early.sh` | `train_earlybalance.tsv` | EarlyPD |
| `run_5seeds_earlypersp.sh` | `train_early_persp.tsv` in a separate private root | EarlyPD |
| `run_5seeds_allPDval.sh` | `train.tsv` | All-stage PD |

The exact validation and test filenames are in the [main guide](../README.md#4-benchmark-experiment-settings).

## 3. Run training

Inspect one launcher, then run all tasks, runs, and folds:

```bash
DRY_RUN=1 SPLITS_ROOT="/path/to/runtime_splits" bash inception_pd/runs/run_5seeds_all.sh
SPLITS_ROOT="/path/to/runtime_splits" CUDA_VISIBLE_DEVICES=0 bash inception_pd/runs/run_5seeds_all.sh
```

For a first run of one fold, set `NUM_RUNS=1`, `TASK_DIR=folds_tsv_DDK_ANALYSIS_PATAKA`, and `FOLD=1`. Use `OUTPUT_ROOT` to select another setting output directory. The default is:

```text
outputs/inception_pd/run_5seeds_all/<task>/run_<1-5>/fold_<1-5>/
```

Each fold selects a checkpoint using validation AUC, tunes a positive-class F1 threshold on validation, and writes test predictions with `probability`, `true_label`, and `predicted_label`. Its text outputs are `Neurovoz_and_PC_GITA_detailed_results.tsv`, `tuned_threshold.json`, `eval_summary.json`, and `train_log.csv`.

## 4. Private track

Preprocess private WAVs and generate their mel JPGs in the same directory layout. Use the [private training-list generator](../reca_pd/README.md#6-add-private-training-data) to create separate `train_early_persp.tsv` files. Check images and train using separate public and private roots:

```bash
python inception_pd/scripts/check_images.py \
  --splits-dir "/path/to/runtime_splits" \
  --train-splits-dir "/path/to/private_training_splits" \
  --setting early_private

SPLITS_ROOT="/path/to/runtime_splits" \
TRAIN_SPLITS_ROOT="/path/to/private_training_splits" \
CUDA_VISIBLE_DEVICES=0 bash inception_pd/runs/run_5seeds_earlypersp.sh
```

## 5. Evaluate results

Run the common benchmark aggregation after a setting finishes:

```bash
SUBJECT_SAMPLE_SALT_ROOT="/data/storage2/gits/CNN-PD-Voice/outputs" \
  ./evals/run_all.sh outputs/inception_pd/run_5seeds_all
```

The `SUBJECT_SAMPLE_SALT_ROOT` string reproduces the historical Sentence k=3/k=10 sampling; the old directory need not exist. Archived text predictions, thresholds, and the regenerated main, gender, and subject-level results are in [`results/inception_pd/experiments/`](../results/inception_pd/experiments/). Their `AUDIOFILE` values record the original server paths; use locally generated `runtime_splits/` when training on another machine. The older `early2` Sentence rerun is excluded from this integration.
