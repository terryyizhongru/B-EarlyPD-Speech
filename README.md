# A Benchmark for Early-stage Parkinson's Disease Detection from Speech

Data preparation and fixed five-fold splits for **A Benchmark for Early-stage Parkinson's Disease Detection from Speech** (Interspeech 2026; arXiv:2605.14066).

This guide assumes you have downloaded **NeuroVoz** and **PC-GITA**, including the PC-GITA metadata. Follow the steps below to prepare audio and obtain training, validation, and test filelists. Run all commands from the repository root in the same Bash terminal.

## Benchmark splits

The repository provides fixed, speaker-independent five-fold splits for three speech tasks:

- DDK /pa-ta-ka/
- Sustained vowel /a/
- Sentence reading

The files under `folds_csv/` define the speaker-level splits. The task-specific TSV files contain the recordings that are available after task selection and audio preprocessing. Validation and test folds target 6 PD and 6 healthy-control speakers per group.

## 1. Install dependencies

Use Python 3.10+ and SoX:

```bash
sudo apt-get install sox python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-preprocess.txt
```

## 2. Standardize filenames

Replace the input paths below with your downloaded dataset directories. For PC-GITA, use the directory directly containing the task folders. Choose a new output directory for this run.

```bash
PD_WORK="/path/to/earlypd-prepared"
PD_DATA="$PD_WORK/NeuroVoz_PCGITA"

python preprocess_scripts/rename_neurovoz.py \
  --data-dir /path/to/NeuroVoz/audios \
  --new-data-dir "$PD_DATA/neurovoz_data/audios"

python preprocess_scripts/rename_restruct_gita.py \
  --data-dir /path/to/PC-GITA \
  --metadata-path /path/to/Copia_de_PCGITA_metadata.xlsx \
  --new-data-dir "$PD_DATA/pcgita_data/audios"
```

These commands copy the audio into a consistent naming scheme and preserve the original files.

## 3. Preprocess audio

Convert both datasets to **mono, 16 kHz, 16-bit WAV**, with **−3 dB peak normalization**:

```bash
python benchmark_tools/prepare_audio.py \
  --wav-dir "$PD_DATA/neurovoz_data/audios" \
  --output-dir "$PD_DATA/neurovoz_data/audios_fortrain"

python benchmark_tools/prepare_audio.py \
  --wav-dir "$PD_DATA/pcgita_data/audios" \
  --output-dir "$PD_DATA/pcgita_data/audios_fortrain"
```

The processed audio is saved in each dataset's `audios_fortrain/` directory. Output directories must not already exist.

## 4. Generate local filelists

Update the supplied splits to point to your processed audio:

```bash
python benchmark_tools/prepare_splits.py \
  --splits-dir benchmark_splits \
  --data-root "$PD_DATA" \
  --output-dir "$PD_WORK/runtime_splits"
```

The new filelists are saved in `runtime_splits/`. Only audio paths change; the benchmark's subjects, recordings, labels, and fold assignments are preserved.

## 5. Check the prepared data

```bash
python benchmark_tools/validate_splits.py \
  --splits-dir "$PD_WORK/runtime_splits" \
  --check-audio \
  --report "$PD_WORK/audit.json"
```

Check that `split_errors` and `audio_errors` are both **0**. If not, inspect `audit.json` before training.

The current splits also produce warnings about missing task recordings and differences in HC training cohorts between settings. These are recorded in the report; the validator does not change the splits.

### Validation notes

Because some task recordings are unavailable after preprocessing, a few validation or test lists contain one or two fewer PD or healthy-control speakers than the nominal 6 PD + 6 HC; these are the lists used in the paper experiments and should be retained for reproduction.

## 6. Use the filelists

Choose a task and a fold (`fold_1` through `fold_5`):

| Task | Directory under `runtime_splits/` |
|---|---|
| DDK /pa-ta-ka/ | `folds_tsv_DDK_ANALYSIS_PATAKA/` |
| Sustained vowel /a/ | `folds_tsv_SUSTAINED-VOWELS_onlyA123/` |
| Sentence reading | `folds_tsv_SENTENCES/` |

Within each fold, select one training setting and use the matching validation and test lists:

| Purpose | File |
|---|---|
| AllPD training | `train_and_val/train.tsv` |
| AllPD-subset training | `train_and_val/train_allPDsubset.tsv` |
| EarlyPD training | `train_and_val/train_earlybalance.tsv` |
| EarlyPD validation | `train_and_val/val_early6PD6HC.tsv` |
| EarlyPD test | `test_early6PD6HC.tsv` |

Each TSV contains `ID` (speaker), `AUDIOFILE` (local WAV path), and `DIAGNOSIS` (`Healthy` or `Parkinson`). Read speaker IDs as strings to preserve leading zeros.

For example, preview the Sentence fold 1 training list:

```bash
head -n 5 "$PD_WORK/runtime_splits/folds_tsv_SENTENCES/fold_1/train_and_val/train.tsv"
```

Model training and unified evaluation instructions will be added separately.


## Acknowledgements

This work is part of the Responsible AI for Voice Diagnostics (RAIVD) project, file number NGF.1607.22.013, under the NGF AiNed Fellowship Grants research program financed by the Dutch Research Council (NWO). It used the Dutch national e-infrastructure with support from SURF Cooperative under grant EINF-10519.


## Citation

```bibtex
@article{zhong2026benchmark,
  title={A Benchmark for Early-stage Parkinson's Disease Detection from Speech},
  author={Zhong, Terry Yi and Tejedor-Garcia, Cristian and Truong, Khiet P and Maas, Janna and ten Bosch, Louis and Bloem, Bastiaan R},
  journal={arXiv preprint arXiv:2605.14066},
  year={2026}
}
```
