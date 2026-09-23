# Benchmark result evaluation

These scripts aggregate one completed experiment setting at a time. They can evaluate RECA-PD or another model that writes the same per-fold prediction format. Run commands from the `PD-early` repository root.

## Required experiment structure

A setting directory must organize results by task, run, and fold:

```text
run_5seeds_all/
├── folds_tsv_DDK_ANALYSIS_PATAKA/
│   ├── run_1/
│   │   ├── fold_1/
│   │   │   ├── Neurovoz_and_PC_GITA_detailed_results.tsv
│   │   │   └── tuned_thresholds.json
│   │   └── ... fold_2 to fold_5
│   └── ... run_2 to run_5
├── folds_tsv_SENTENCES/
│   └── ... the same run/fold structure
└── folds_tsv_SUSTAINED-VOWELS_onlyA123/
    └── ... the same run/fold structure
```

Each task must contain `run_1` through `run_5`, with `fold_1` through `fold_5` under every run.

The detailed TSV contains one row per test recording:

```text
ID  AUDIOFILE  DIAGNOSIS  probability  true_label  predicted_label
```

- `ID` is the speaker ID and must be read as a string.
- `AUDIOFILE` identifies the evaluated recording.
- `DIAGNOSIS` is `Healthy` or `Parkinson`.
- `probability` is the model score for the positive PD class.
- `true_label` and `predicted_label` are binary integers, where PD is 1.
- `predicted_label` is obtained by comparing `probability` with the threshold tuned on that fold's validation data.

Sentence and Vowel subject-level evaluation also require a threshold file in each fold. Both of these formats are supported:

```json
{
  "thresholds": {
    "Neurovoz_and_PC_GITA": 0.37
  }
}
```

```json
{
  "threshold": 0.37
}
```

## Export predictions from another model

Use `BenchmarkResultWriter` after each task/run/fold inference. Its test prediction input is a TSV with these columns:

```text
ID  AUDIOFILE  DIAGNOSIS  probability  true_label  predicted_label
31  /data/example.wav  Parkinson  0.82  1  1
```

The first five columns are required. `predicted_label` is optional; the writer generates it from the fold threshold when omitted and validates it when supplied. `probability` must be the positive-class probability. Preserve speaker IDs as strings.

The threshold must be selected on validation data. To use the RECA-PD rule, supply a validation TSV containing `probability` and `true_label`:

```bash
python evals/benchmark_output.py /path/to/test_predictions.tsv \
  --setting-dir /path/to/run_5seeds_all \
  --task sentence \
  --run 1 \
  --fold 1 \
  --validation-predictions /path/to/validation_predictions.tsv
```

The tool searches 101 thresholds from 0 to 1 and selects the one with maximum positive-class F1. If the model has its own validation threshold rule, provide the resulting value instead:

```bash
python evals/benchmark_output.py /path/to/test_predictions.tsv \
  --setting-dir /path/to/run_5seeds_all \
  --task sentence \
  --run 1 \
  --fold 1 \
  --threshold 0.37
```

Task aliases are `ddk`, `vowel`, and `sentence`; a complete `folds_tsv_*` directory name is also accepted. The writer automatically creates the task/run/fold directory and writes `Neurovoz_and_PC_GITA_detailed_results.tsv` plus `tuned_thresholds.json`. Existing outputs are protected unless `--overwrite` is supplied. Repeat this call for every task, run, and fold.

To integrate the writer directly into a model pipeline:

```python
from evals.benchmark_output import BenchmarkResultWriter

writer = BenchmarkResultWriter("/path/to/run_5seeds_all")
threshold = writer.select_threshold(validation_rows)
result_tsv, threshold_json = writer.write_fold(
    task="sentence",
    run=1,
    fold=1,
    records=test_rows,
    threshold=threshold,
    threshold_strategy="max_pos_f1_on_validation",
    threshold_grid=101,
)
```

`validation_rows` needs `probability` and `true_label`. Each item in `test_rows` is a mapping with the same fields as one test TSV row.

## Aggregate one setting

Specify exactly one setting directory:

```bash
./evals/run_all.sh \
  results/reca_pd/experiments/run_5seeds_all
```

If NumPy and scikit-learn are installed in another environment:

```bash
PYTHON_BIN=/path/to/python \
  ./evals/run_all.sh results/reca_pd/experiments/run_5seeds_all
```

The setting directories used by the paper are:

| Directory | Experiment |
|---|---|
| `run_5seeds_all` | AllPD |
| `run_5seeds_allsubset` | AllPD-subset |
| `run_5seeds_early` | EarlyPD only |
| `run_5seeds_earlypersp` | EarlyPD plus private prospective data |
| `run_5seeds_allPDval` | All-stage validation/test experiment |

The three Python aggregators accept either the complete setting directory shown above or one `folds_tsv_*` task directory. Sentence k=3/k=10 sampling uses a stable string built from a logical experiment root, setting, task, run, fold, and speaker ID. The default root reproduces the historical RECA-PD sampling. To reproduce the archived BDHPD or Inception-PD subject-level numbers, set `SUBJECT_SAMPLE_SALT_ROOT` when using `run_all.sh` (or pass `--sample-salt-root` to `aggregate_subject_results.py`):

```bash
SUBJECT_SAMPLE_SALT_ROOT=/data/storage1t/projects/early/BDHPD \
  ./evals/run_all.sh results/bdhpd/experiments/run_5seeds_all

SUBJECT_SAMPLE_SALT_ROOT=/data/storage2/gits/CNN-PD-Voice/outputs \
  ./evals/run_all.sh results/inception_pd/experiments/run_5seeds_all
```

These strings seed the sampler; the original directories do not need to exist on the machine running evaluation. Use the same value across repeated evaluations of one model.

To process every task in a setting:

```bash
python evals/aggregate_main_results.py /path/to/run_5seeds_all
python evals/aggregate_gender_results.py /path/to/run_5seeds_all
python evals/aggregate_subject_results.py /path/to/run_5seeds_all
```

To process only Sentence, for example:

```bash
TASK_DIR=/path/to/run_5seeds_all/folds_tsv_SENTENCES
python evals/aggregate_main_results.py "$TASK_DIR"
python evals/aggregate_gender_results.py "$TASK_DIR"
python evals/aggregate_subject_results.py "$TASK_DIR"
```

The output is written into the selected task directory in both modes. Subject-level aggregation produces files only for Sentence and Sustained Vowels; an explicitly selected DDK task is skipped. The gender script uses `evals/metadata/speaker_gender.csv` by default. Use `--meta-csv PATH` to provide another file containing `ID` and `Gender` columns.

## Generated results

Files are written directly into each task directory:

```text
run_5seeds_all/
├── folds_tsv_DDK_ANALYSIS_PATAKA/
│   ├── average_runs_folds.txt
│   └── average_gender_runs_folds.txt
├── folds_tsv_SENTENCES/
│   ├── average_runs_folds.txt
│   ├── average_gender_runs_folds.txt
│   ├── average_subjectlevel_sentences_k3.tsv
│   └── average_subjectlevel_sentences_k10.tsv
└── folds_tsv_SUSTAINED-VOWELS_onlyA123/
    ├── average_runs_folds.txt
    ├── average_gender_runs_folds.txt
    └── average_subjectlevel_vowels.tsv
```

### Main result

`average_runs_folds.txt` contains only final aggregates and has no individual run tables:

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

`average_gender_runs_folds.txt` reports final F1 and AUC for male and female speakers:

```text
Gender  Mean_F1  RunStd_F1  MeanFoldStd_F1  Mean_AUC  RunStd_AUC  MeanFoldStd_AUC
M       0.6912   0.0258     0.1080          0.7554    0.0154      0.1259
F       0.7044   0.0323     0.1540          0.7717    0.0196      0.1457
```

### Subject-level result

For Sentence, `average_subjectlevel_sentences_k3.tsv` and `average_subjectlevel_sentences_k10.tsv` sample 3 or 10 recordings per speaker, average their probabilities, and apply the fold threshold.

For Vowel, `average_subjectlevel_vowels.tsv` averages all available sustained-vowel probabilities for each speaker and applies the fold threshold.

```text
root  runs  sample_k  sample_seed  Mean_F1  Mean_AUC  Delta_F1  Delta_AUC
folds_tsv_SUSTAINED-VOWELS_onlyA123  run_1,...,run_5  0  0  0.6224  0.5976  -0.0291  -0.0283
```

`Delta_F1` and `Delta_AUC` are measured against the recording-level result obtained with the same saved fold thresholds. `sample_k=0` means all available recordings per speaker are used.

## Reproduce the paper result tables

1. Train five runs and five folds for one setting.
2. Save every task/run/fold prediction TSV and tuned validation threshold in the required structure.
3. Run `evals/run_all.sh SETTING_DIR`.
4. Read the primary benchmark values from the `Combined` rows of `average_runs_folds.txt`.
5. Use the `PC-GITA` and `NeuroVoz` rows for dataset-specific results.
6. Use `average_gender_runs_folds.txt` for gender analysis and the Sentence/Vowel subject-level TSVs for speaker-level analysis.
