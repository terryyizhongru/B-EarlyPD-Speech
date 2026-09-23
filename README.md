# A Benchmark for Early-stage Parkinson's Disease Detection from Speech

This repository contains the official benchmark splits and evaluation protocols for the paper:
**“A Benchmark for Early-stage Parkinson's Disease Detection from Speech”**

[📄 Paper](https://arxiv.org/abs/2605.14066) | [📖 Citation](#citation) | 📣 *Accepted for [Interspeech 2026](https://arxiv.org/abs/2605.14066)!*

---

## Acknowledgements

This work is part of the Responsible AI for Voice Diagnostics (RAIVD) project, file number NGF.1607.22.013, under the NGF AiNed Fellowship Grants research program financed by the Dutch Research Council (NWO). It used the Dutch national e-infrastructure with support from SURF Cooperative under grant EINF-10519.

## Latest Updates

- Fixed speaker-independent five-fold splits are available for DDK, sustained vowel, and Sentence tasks.
- Unified prediction export and evaluation tools are available for recording-, gender-, and subject-level results.
- RECA-PD, BDHPD, and Inception-PD reproduction guides and archived text results are available.

---

## Contents

- [Benchmark splits](#benchmark-splits)
- [1. Install dependencies](#1-install-dependencies)
- [2. Prepare audio](#2-prepare-audio)
- [3. Generate and validate local filelists](#3-generate-and-validate-local-filelists)
- [4. Benchmark experiment settings](#4-benchmark-experiment-settings)
- [5. Run your model](#5-run-your-model)
- [6. Export and evaluate model predictions](#6-export-and-evaluate-model-predictions)
- [7. Reproduce baseline models from the paper](#7-reproduce-baseline-models-from-the-paper)
- [Citation](#citation)

This guide assumes you have downloaded **NeuroVoz** and **PC-GITA**, including the PC-GITA metadata. Follow the steps below to prepare audio and obtain training, validation, and test filelists. Run all commands from the repository root in the same Bash terminal.

## Benchmark splits

The benchmark provides fixed, speaker-independent five-fold splits for DDK /pa-ta-ka/, sustained vowel /a/, and sentence reading. Each task has recording-level training, validation, and test TSVs based on the speaker-level fold definitions.

See [Benchmark experiment settings](#4-benchmark-experiment-settings) for the task directories, fold layout, TSV columns, and lists used by each paper setting.

## 1. Install dependencies

Use Python 3.10+ and SoX:

```bash
sudo apt-get install sox python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-preprocess.txt
```

## 2. Prepare audio

`PD_WORK` is the root of a new prepared-data workspace on your machine. It is where these commands write renamed audio, processed audio, local filelists, and validation reports; it is not one of the downloaded datasets. Choose any writable location with enough space. `PD_DATA` is the dataset directory created inside that workspace:

```bash
PD_WORK="/path/to/earlypd-prepared"
PD_DATA="$PD_WORK/NeuroVoz_PCGITA"
```

The completed preparation will have this layout:

```text
earlypd-prepared/
├── NeuroVoz_PCGITA/
│   ├── neurovoz_data/
│   └── pcgita_data/
├── runtime_splits/
└── audit.json
```

### Standardize filenames

Replace the input paths below with the downloaded dataset directories. For PC-GITA, use the directory directly containing its task folders.

```bash
python preprocess_scripts/rename_neurovoz.py \
  --data-dir /path/to/NeuroVoz/audios \
  --new-data-dir "$PD_DATA/neurovoz_data/audios"

python preprocess_scripts/rename_restruct_gita.py \
  --data-dir /path/to/PC-GITA \
  --metadata-path /path/to/Copia_de_PCGITA_metadata.xlsx \
  --new-data-dir "$PD_DATA/pcgita_data/audios"
```

These commands copy the audio into a consistent naming scheme and preserve the original files.

### Preprocess audio

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

## 3. Generate and validate local filelists

Generate a local copy of the fixed splits whose `AUDIOFILE` values point to the prepared audio:

```bash
python benchmark_tools/prepare_splits.py \
  --splits-dir benchmark_splits \
  --data-root "$PD_DATA" \
  --output-dir "$PD_WORK/runtime_splits"
```

Only the audio paths change; subjects, recordings, labels, and fold assignments are preserved. Validate the generated lists and WAV files before running a model:

```bash
python benchmark_tools/validate_splits.py \
  --splits-dir "$PD_WORK/runtime_splits" \
  --check-audio \
  --report "$PD_WORK/audit.json"
```

Check that `split_errors` and `audio_errors` are both **0**. If not, inspect `audit.json` before training. The report can also contain warnings about unavailable task recordings and differences in HC training cohorts between settings; the validator records these conditions without changing the splits.

### Validation notes

Because some task recordings are unavailable after preprocessing, a few validation or test lists contain one or two fewer PD or healthy-control speakers than the nominal 6 PD + 6 HC; these are the lists used in the paper experiments and should be retained for reproduction.

## 4. Benchmark experiment settings

The benchmark contains three task directories under `runtime_splits/`:

| Task | Directory |
|---|---|
| DDK /pa-ta-ka/ | `folds_tsv_DDK_ANALYSIS_PATAKA/` |
| Sustained vowel /a/ | `folds_tsv_SUSTAINED-VOWELS_onlyA123/` |
| Sentence reading | `folds_tsv_SENTENCES/` |

`benchmark_splits/folds_csv/` defines the speaker-level folds. Each task directory contains `fold_1/` through `fold_5/`. Within each fold, training and validation TSVs are in `train_and_val/`, while test TSVs are directly under `fold_N/`. The task TSVs contain recordings available after task selection and preprocessing.

The paper experiments include the following training settings.

| Setting | Training list | Validation list | Test list |
|---|---|---|---|
| AllPD | `train_and_val/train.tsv` | `train_and_val/val_early6PD6HC.tsv` | `test_early6PD6HC.tsv` |
| AllPD-subset | `train_and_val/train_allPDsubset.tsv` | `train_and_val/val_early6PD6HC.tsv` | `test_early6PD6HC.tsv` |
| EarlyPD | `train_and_val/train_earlybalance.tsv` | `train_and_val/val_early6PD6HC.tsv` | `test_early6PD6HC.tsv` |
| EarlyPD + private data (Private track) | `train_and_val/train_early_persp.tsv` in a separate private training root | `train_and_val/val_early6PD6HC.tsv` | `test_early6PD6HC.tsv` |
| All-stage validation/test | `train_and_val/train.tsv` | `train_and_val/val_all6PD6HC.tsv` | `test_all6PD6HC.tsv` |

The public preparation steps generate the public lists. The Private track training list is model-specific and is created separately while keeping public validation and test lists unchanged.

Each TSV contains `ID` (speaker), `AUDIOFILE` (local WAV path), and `DIAGNOSIS` (`Healthy` or `Parkinson`). Read speaker IDs as strings to preserve leading zeros. For example:

```bash
head -n 5 "$PD_WORK/runtime_splits/folds_tsv_SENTENCES/fold_1/train_and_val/train.tsv"
```

## 5. Run your model

Use the fixed filelists to train and test your model for each speech task, run, and fold. Within each fold:

1. Train on the selected training TSV.
2. Use only the matching validation TSV for checkpoint selection, hyperparameter selection, and the decision threshold.
3. Run the selected model once on the matching test TSV.
4. Keep the positive-class probability and binary label for every test recording.

The paper protocol uses five independent runs of all five folds. A model may use its own architecture and features, but it must preserve the supplied speaker splits and must not use the test labels for model or threshold selection. The next section converts its predictions to the common evaluation format.

For a complete reference implementation, see the [RECA-PD training guide](reca_pd/README.md#5-run-training). Its launchers show how to run three tasks, five folds, and five independent runs, export prediction scores, and add private training data.

## 6. Export and evaluate model predictions

The evaluation code can aggregate predictions from RECA-PD or another model. After completing one experimental setting, organize its outputs by task, run, and fold:

```text
run_5seeds_all/
├── folds_tsv_DDK_ANALYSIS_PATAKA/
│   └── run_1 ... run_5/
│       └── fold_1 ... fold_5/
├── folds_tsv_SENTENCES/
│   └── run_1 ... run_5/
│       └── fold_1 ... fold_5/
└── folds_tsv_SUSTAINED-VOWELS_onlyA123/
    └── run_1/
        └── fold_1/
            ├── Neurovoz_and_PC_GITA_detailed_results.tsv
            └── tuned_thresholds.json
```

Every task should contain five runs, and every run should contain five folds. The detailed TSV stores one test recording per row:

```text
ID  AUDIOFILE  DIAGNOSIS  probability  true_label  predicted_label
```

`probability` is the model score for the positive PD class. `predicted_label` is obtained by comparing that probability with the threshold tuned on the fold's validation data. Keep the threshold JSON in the same fold directory for Sentence and Vowel subject-level evaluation.

### Export results from another model

For each task, run, and fold, save the test predictions as a TSV:

```text
ID  AUDIOFILE  DIAGNOSIS  probability  true_label  predicted_label
31  /data/example.wav  Parkinson  0.82  1  1
```

`predicted_label` is optional. When supplied, the exporter checks that it agrees with `probability >= threshold`; otherwise it generates the label. Select the threshold using validation predictions, never the test labels.

The result tool creates the required task/run/fold directories and both benchmark files. It can select the threshold from a validation TSV containing `probability` and `true_label`:

```bash
python evals/benchmark_output.py /path/to/test_predictions.tsv \
  --setting-dir /path/to/run_5seeds_all \
  --task sentence \
  --run 1 \
  --fold 1 \
  --validation-predictions /path/to/validation_predictions.tsv
```

This uses the same 101-point maximum positive-class F1 rule as RECA-PD. If the model already selected a validation threshold, replace `--validation-predictions` with `--threshold 0.37`. Repeat the call for every task/run/fold. The tool writes:

```text
run_5seeds_all/
└── folds_tsv_SENTENCES/
    └── run_1/
        └── fold_1/
            ├── Neurovoz_and_PC_GITA_detailed_results.tsv
            └── tuned_thresholds.json
```

The same tool can be called from Python after inference:

```python
from evals.benchmark_output import BenchmarkResultWriter

writer = BenchmarkResultWriter("/path/to/run_5seeds_all")
threshold = writer.select_threshold(validation_rows)
writer.write_fold(
    task="sentence",
    run=1,
    fold=1,
    records=test_rows,
    threshold=threshold,
    threshold_strategy="max_pos_f1_on_validation",
    threshold_grid=101,
)
```

After exporting all tasks, runs, and folds for one setting, generate the final paper-format results:

```bash
./evals/run_all.sh /path/to/run_5seeds_all
```

This produces compact main results for the combined datasets and for PC-GITA and NeuroVoz separately, gender-level results, Sentence subject-level k3/k10 results, and Vowel subject-level results.

### Run one evaluation level

Each Python aggregator accepts either the complete setting directory or one task directory. Use the setting directory to process all tasks:

```bash
python evals/aggregate_main_results.py /path/to/run_5seeds_all
python evals/aggregate_gender_results.py /path/to/run_5seeds_all
python evals/aggregate_subject_results.py /path/to/run_5seeds_all
```

Or process only one task:

```bash
python evals/aggregate_main_results.py /path/to/run_5seeds_all/folds_tsv_SENTENCES
python evals/aggregate_gender_results.py /path/to/run_5seeds_all/folds_tsv_SENTENCES
python evals/aggregate_subject_results.py /path/to/run_5seeds_all/folds_tsv_SENTENCES
```

Results are written into the selected task directory. Subject-level aggregation applies only to Sentence and Sustained Vowels. The gender script uses `evals/metadata/speaker_gender.csv` by default. Use `--meta-csv PATH` to provide another file containing `ID` and `Gender` columns.

### Main result

Each task receives an `average_runs_folds.txt` containing only final aggregates, without individual run tables:

```text
Cohort    Metric    Mean    RunStd    MeanFoldStd
Combined  F1        0.7051  0.0157    0.0747
PC-GITA   F1        0.7263  0.0280    0.0851
NeuroVoz  F1        0.6970  0.0139    0.0909
```

The complete file reports Accuracy, F1, Precision, AUC, Sensitivity, Specificity, and Balanced Accuracy for every cohort.

- `Combined` is the overall NeuroVoz + PC-GITA result used as the main benchmark result.
- `PC-GITA` and `NeuroVoz` report the same trained model on each dataset separately.
- `Mean` first averages five folds within each run and then averages the five run means.
- `RunStd` is the population standard deviation across the five run means.
- `MeanFoldStd` is the mean of the within-run fold standard deviations.

Label metrics use the saved `predicted_label`. AUC uses the positive-class score in `probability`.

### Gender-level result

Each task receives an `average_gender_runs_folds.txt` reporting final F1 and AUC for male and female speakers:

```text
Gender  Mean_F1  RunStd_F1  MeanFoldStd_F1  Mean_AUC  RunStd_AUC  MeanFoldStd_AUC
M       0.6912   0.0258     0.1080          0.7554    0.0154      0.1259
F       0.7044   0.0323     0.1540          0.7717    0.0196      0.1457
```

### Subject-level result

For Sentence, `average_subjectlevel_sentences_k3.tsv` and `average_subjectlevel_sentences_k10.tsv` sample 3 or 10 recordings per speaker, average their probabilities, and apply the fold threshold.

For Vowel, `average_subjectlevel_vowels.tsv` averages all available sustained-vowel probabilities for each speaker and applies the fold threshold:

```text
root  runs  sample_k  sample_seed  Mean_F1  Mean_AUC  Delta_F1  Delta_AUC
folds_tsv_SUSTAINED-VOWELS_onlyA123  run_1,...,run_5  0  0  0.6224  0.5976  -0.0291  -0.0283
```

`Delta_F1` and `Delta_AUC` are measured against the recording-level result obtained with the same saved fold thresholds. `sample_k=0` means all available recordings per speaker are used.

The complete directory contract and additional examples are documented in [`evals/README.md`](evals/README.md). Archived RECA-PD examples are available under [`results/reca_pd/experiments/`](results/reca_pd/experiments/).

## 7. Reproduce baseline models from the paper

Prepare `runtime_splits/` with the steps above, then follow the model-specific guide for any additional preprocessing, five-run/five-fold training, and evaluation. The baseline code in this repository was adapted from the official code repositories for each method:

- RECA-PD: [benchmark reproduction guide](reca_pd/README.md) · [official source code](https://github.com/terryyizhongru/RECA-PD)
- BDHPD: [benchmark reproduction guide](bdhpd/README.md) · [official source code](https://github.com/MorenoLaQuatra/BDHPD)
- Inception-PD: [benchmark reproduction guide](inception_pd/README.md) · [official source code](https://github.com/terryyizhongru/CNN-PD-Voice)

Archived text results are under `results/reca_pd/`, `results/bdhpd/`, and `results/inception_pd/`.

### Add private training data (Private track)

A private cohort can be added to the training portion of a baseline experiment while keeping the public benchmark validation and test sets unchanged. At a high level:

1. Prepare the private recordings and any features required by the model.
2. Create task-specific private TSVs containing `ID`, `AUDIOFILE`, and `DIAGNOSIS`.
3. Append those records to the selected training list in every fold, writing the augmented lists under a separate training-split root.
4. Use the public `runtime_splits/` for validation and test, and the augmented root only for training.

Private speaker IDs must be unique and must not overlap any public benchmark speaker. Results obtained with a different private cohort follow the same protocol but are not an exact reproduction of the paper's private-data result. See [Add private training data with RECA-PD](reca_pd/README.md#6-add-private-training-data) for a concrete generator, validation command, and launcher example.



## Citation

```bibtex
@article{zhong2026benchmark,
  title={A Benchmark for Early-stage Parkinson's Disease Detection from Speech},
  author={Zhong, Terry Yi and Tejedor-Garcia, Cristian and Truong, Khiet P and Maas, Janna and ten Bosch, Louis and Bloem, Bastiaan R},
  journal={arXiv preprint arXiv:2605.14066},
  year={2026}
}
```
