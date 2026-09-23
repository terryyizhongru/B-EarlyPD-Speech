#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {all|all_subset|early|early_private|all_stage}" >&2
  exit 2
fi

regime="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
splits_root="${SPLITS_ROOT:-${repo_root}/benchmark_splits}"
train_splits_root="${TRAIN_SPLITS_ROOT:-${splits_root}}"
python_bin="${PYTHON:-python3}"
num_runs="${NUM_RUNS:-5}"
base_seed="${BASE_SEED:-42}"
dry_run="${DRY_RUN:-0}"
cuda_device="${CUDA_VISIBLE_DEVICES:-0}"
epochs="${EPOCHS:-20}"
patience="${PATIENCE:-5}"
min_epochs="${MIN_EPOCHS:-5}"
batch_size="${BATCH_SIZE:-4}"
img_size="${IMG_SIZE:-600}"

tasks=(folds_tsv_DDK_ANALYSIS_PATAKA folds_tsv_SUSTAINED-VOWELS_onlyA123 folds_tsv_SENTENCES)
case "$regime" in
  all) setting=run_5seeds_all; train_file=train.tsv; val_file=val_early6PD6HC.tsv; test_file=test_early6PD6HC.tsv ;;
  all_subset) setting=run_5seeds_allsubset; train_file=train_allPDsubset.tsv; val_file=val_early6PD6HC.tsv; test_file=test_early6PD6HC.tsv ;;
  early) setting=run_5seeds_early; train_file=train_earlybalance.tsv; val_file=val_early6PD6HC.tsv; test_file=test_early6PD6HC.tsv ;;
  early_private)
    setting=run_5seeds_earlypersp; train_file=train_early_persp.tsv; val_file=val_early6PD6HC.tsv; test_file=test_early6PD6HC.tsv
    tasks=(folds_tsv_DDK_ANALYSIS_PATAKA folds_tsv_SUSTAINED-VOWELS_onlyA123)
    if [[ "$train_splits_root" == "$splits_root" ]]; then
      echo "Private training requires TRAIN_SPLITS_ROOT; public validation and test remain under SPLITS_ROOT." >&2
      exit 2
    fi
    ;;
  all_stage) setting=run_5seeds_allPDval; train_file=train.tsv; val_file=val_all6PD6HC.tsv; test_file=test_all6PD6HC.tsv ;;
  *) echo "Unknown regime: $regime" >&2; exit 2 ;;
esac

output_root="${OUTPUT_ROOT:-${repo_root}/outputs/inception_pd/${setting}}"

split_dir() {
  local root="$1" task="$2" fold="$3"
  if [[ -d "$root/$task/fold_$fold/train_and_val" ]]; then
    printf '%s\n' "$root/$task/fold_$fold/train_and_val"
  elif [[ -d "$root/$task/fold_$fold/sub_splits" ]]; then
    printf '%s\n' "$root/$task/fold_$fold/sub_splits"
  else
    echo "Missing train/validation directory: $root/$task/fold_$fold" >&2
    return 1
  fi
}

cd "$repo_root"
for task in "${tasks[@]}"; do
  if [[ -n "${TASK_DIR:-}" && "$task" != "$TASK_DIR" ]]; then continue; fi
  for run_idx in $(seq 1 "$num_runs"); do
    seed=$((base_seed + run_idx - 1))
    for fold in 1 2 3 4 5; do
      if [[ -n "${FOLD:-}" && "$fold" != "$FOLD" ]]; then continue; fi
      train_dir="$(split_dir "$train_splits_root" "$task" "$fold")"
      val_dir="$(split_dir "$splits_root" "$task" "$fold")"
      train_tsv="$train_dir/$train_file"
      val_tsv="$val_dir/$val_file"
      test_tsv="$splits_root/$task/fold_$fold/$test_file"
      fold_output="$output_root/$task/run_$run_idx/fold_$fold"
      for required in "$train_tsv" "$val_tsv" "$test_tsv"; do
        if [[ ! -f "$required" ]]; then echo "Missing required file: $required" >&2; exit 1; fi
      done
      command=("$python_bin" inception_pd/Inception_pd_detection_voice.py
        --train_tsv "$train_tsv" --val_tsv "$val_tsv" --test_tsv "$test_tsv"
        --output_dir "$fold_output" --epochs "$epochs"
        --early_stop_patience "$patience" --min_epochs "$min_epochs"
        --batch_size "$batch_size" --img_size "$img_size" --seed "$seed")
      printf '[Inception-PD:%s] task=%s run=%s fold=%s seed=%s\n' "$regime" "$task" "$run_idx" "$fold" "$seed"
      if [[ "$dry_run" == 1 ]]; then
        printf '  '; printf '%q ' env "CUDA_VISIBLE_DEVICES=$cuda_device" "${command[@]}"; printf '\n'
      else
        mkdir -p "$fold_output"
        CUDA_VISIBLE_DEVICES="$cuda_device" "${command[@]}"
      fi
    done
  done
done
