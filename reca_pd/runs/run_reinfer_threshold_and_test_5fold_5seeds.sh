#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

splits_root="${SPLITS_ROOT:-${repo_root}/benchmark_splits}"
config_path="${CONFIG_PATH:-${repo_root}/reca_pd/configs/framework.yaml}"
checkpoint_root="${CHECKPOINT_ROOT:-${repo_root}/outputs/reca_pd/all}"
python_bin="${PYTHON:-python3}"
device="${DEVICE:-cuda}"
cuda_device="${CUDA_VISIBLE_DEVICES:-0}"
num_runs="${NUM_RUNS:-5}"
base_seed="${BASE_SEED:-42}"
dry_run="${DRY_RUN:-0}"

taskset=(
    "folds_tsv_DDK_ANALYSIS_PATAKA"
    "folds_tsv_SUSTAINED-VOWELS_onlyA123"
    "folds_tsv_SENTENCES"
)

split_dir() {
    local task_dir="$1"
    local fold="$2"
    if [[ -d "${splits_root}/${task_dir}/fold_${fold}/train_and_val" ]]; then
        printf '%s\n' "${splits_root}/${task_dir}/fold_${fold}/train_and_val"
    elif [[ -d "${splits_root}/${task_dir}/fold_${fold}/sub_splits" ]]; then
        printf '%s\n' "${splits_root}/${task_dir}/fold_${fold}/sub_splits"
    else
        echo "Missing train/validation directory for ${task_dir}/fold_${fold}" >&2
        return 1
    fi
}

task_filter() {
    case "$1" in
        folds_tsv_DDK_ANALYSIS_PATAKA) printf '%s\n' "DDK" ;;
        folds_tsv_SUSTAINED-VOWELS_onlyA123) printf '%s\n' "SUSTAINED-VOWELS" ;;
        folds_tsv_SENTENCES) printf '%s\n' "SENTENCES" ;;
        *) echo "Unknown task directory: $1" >&2; return 1 ;;
    esac
}

cd "${repo_root}"

for task_dir in "${taskset[@]}"; do
    filter_task="$(task_filter "${task_dir}")"
    for run_idx in $(seq 1 "${num_runs}"); do
        seed=$((base_seed + run_idx - 1))
        for fold in 1 2 3 4 5; do
            lists_dir="$(split_dir "${task_dir}" "${fold}")"
            train_tsv="${lists_dir}/train.tsv"
            val_tsv="${lists_dir}/val_all6PD6HC.tsv"
            test_tsv="${splits_root}/${task_dir}/fold_${fold}/test_all6PD6HC.tsv"
            output_dir="${checkpoint_root}/${task_dir}/run_${run_idx}/fold_${fold}"

            for required_file in "${train_tsv}" "${val_tsv}" "${test_tsv}" "${config_path}"; do
                if [[ ! -f "${required_file}" ]]; then
                    echo "Missing required file: ${required_file}" >&2
                    exit 1
                fi
            done

            if [[ "${dry_run}" == "1" ]]; then
                checkpoint_path="${output_dir}/model_checkpoints/best_epoch_*.pth"
            else
                mapfile -t checkpoints < <(
                    compgen -G "${output_dir}/model_checkpoints/best_epoch_*.pth" | sort
                )
                if [[ ${#checkpoints[@]} -ne 1 ]]; then
                    echo "Expected one best checkpoint in ${output_dir}; found ${#checkpoints[@]}" >&2
                    exit 1
                fi
                checkpoint_path="${checkpoints[0]}"
            fi

            command=(
                "${python_bin}" "reca_pd/scripts/model_pipeline/reinfer_threshold_and_test.py"
                --config "${config_path}"
                --training-dataset "${train_tsv}"
                --validation-dataset "${val_tsv}"
                --test-dataset "${test_tsv}"
                --load-checkpoint "${checkpoint_path}"
                --output-dir "${output_dir}"
                --seed "${seed}"
                --yaml-overrides "device:${device}" "seed:${seed}" "model:RECAPD"
                --filter-tasks "${filter_task}"
            )

            printf '[reinfer-all-stage] task=%s run=%s fold=%s seed=%s\n'                 "${filter_task}" "${run_idx}" "${fold}" "${seed}"
            if [[ "${dry_run}" == "1" ]]; then
                printf '  '
                printf '%q ' env "CUDA_VISIBLE_DEVICES=${cuda_device}" "${command[@]}"
                printf '\n'
            else
                CUDA_VISIBLE_DEVICES="${cuda_device}" "${command[@]}"
            fi
        done
    done
done
