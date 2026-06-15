# A Benchmark for Early-stage Parkinson's Disease Detection from Speech

This repository contains the official benchmark splits and evaluation protocols for the paper:  
**"A Benchmark for Early-stage Parkinson's Disease Detection from Speech"**

[📖 Citation](#citation) |
📣 _Accepted for [Interspeech 2026](https://arxiv.org/abs/2605.14066)!_

---

## Acknowledgements

Part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research program NGF AiNed Fellowship Grants, which is financed by the Dutch Research Council (NWO). This work used the Dutch national e-infrastructure with the support of the SURF Cooperative using grant no. EINF-10519.

## Latest Updates
- **Benchmark splits:** Speaker-independent, standardized splits are available in `benchmark_splits/`.
- **Preprocess scripts:** Work in progress; code will be released/cleaned up.
- **Evaluation scripts:** code will be released/cleaned up.

---

## Benchmark splits: directory layout and naming

The `benchmark_splits/` folder contains multiple split groups. Each group is a 5-fold cross-validation setup with:

- `fold_1/` ... `fold_5/`
- One test set per fold (early-stage PD benchmark)
- A `train_and_val/` subfolder per fold containing the training and validation lists

At a high level:

- `folds_csv/`
	- Speaker-level lists with metadata for each subject.
	- Typical columns include: `ID`, `Group/Diagnosis`, demographics, clinical scores (e.g., HY stage, UPDRS), and dataset source.
- `folds_tsv*/`
	- Utterance/audio-level lists used for model training.
	- Each row corresponds to one audio file and includes: subject `ID`, `AUDIOFILE` (wav path), and `DIAGNOSIS` (label).

### What you find inside each fold

Each `fold_x/` directory follows the same naming pattern (extension is `.csv` under `folds_csv/`, and `.tsv` under `folds_tsv*/`):

- Test set:
	- `test_early6PD6HC.*` (early-stage PD benchmark)
	- `test_all6PD6HC.*` (all-stage PD test set)


- Train/validation split (inside the subfolder):
	- `train_and_val/`
		- `train.*` (AllPD setting in the paper: the full set of PD speakers across all stages from the benchmark datasets)
		- validation lists such as `val_early6PD6HC.*` and `val_all6PD6HC.*`

### Split groups (tasks)

- `folds_tsv_all/`
	- Combined list covering all tasks/audios.

Single-task training lists used in the benchmark paper experiments:

- `folds_tsv_SENTENCES/`
- `folds_tsv_DDK_ANALYSIS_PATAKA/`
- `folds_tsv_SUSTAINED-VOWELS_onlyA123/`

For these three single-task split groups, the `train_and_val/` folder also includes:

- `train_allPDsubset.tsv` (AllPD-subset introduced in the paper: a distribution-matched subset of AllPD)
- `train_earlybalance.tsv` (EarlyPD setting: the full set of EarlyPD speakers from the benchmark datasets)

## Preprocessing

Use the helper scripts in `preprocess_scripts/` to normalize raw NeuroVoz/PC-GITA audios before running any benchmark experiments. Provide `$DATASET_DIR` as the parent containing the source downloads and target root for the reorganized files.

1. Rename and reorganize both datasets into `NeuroVoz_PCGITA` (the destination tree used by the benchmark):

```
python3 preprocess_scripts/rename_neurovoz.py $DATASET_DIR/neurovoz_v3/audios $DATASET_DIR/NeuroVoz_PCGITA/neurovoz_data/audios
python3 preprocess_scripts/rename_restruct_gita.py --data-dir $DATASET_DIR/PC-GITA/ --new-data-dir $DATASET_DIR/NeuroVoz_PCGITA/pcgita_data/audios
```

2. Downsample and normalize the reorganized WAVs (default output goes into each `<wav-dir>/audios_fortrain`):

```
python3 preprocess_scripts/wav_preprocessing.py --wav-dir $DATASET_DIR/NeuroVoz_PCGITA/pcgita_data/audios
python3 preprocess_scripts/wav_preprocessing.py --wav-dir $DATASET_DIR/NeuroVoz_PCGITA/neurovoz_data/audios
```

---

## <a name="citation"></a> 📖 Citation



If you use this benchmark or splits from this project in academic work, please cite:

 

### 📄 LaTeX (BibTeX)

```bibtex

@article{zhong2026benchmark,

  title={A Benchmark for Early-stage Parkinson's Disease Detection from Speech},

  author={Zhong, Terry Yi and Tejedor-Garcia, Cristian and Truong, Khiet P and Maas, Janna and Bosch, Louis ten and Bloem, Bastiaan R},

  journal={arXiv preprint arXiv:2605.14066},

  year={2026}

}

```






