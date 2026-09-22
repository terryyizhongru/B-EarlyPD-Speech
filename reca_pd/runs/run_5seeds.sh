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
config_path="${CONFIG_PATH:-${repo_root}/reca_pd/configs/framework.yaml}"
python_bin="${PYTHON:-python3}"
device="${DEVICE:-cuda}"
cuda_device="${CUDA_VISIBLE_DEVICES:-0}"
num_runs="${NUM_RUNS:-5}"
dry_run="${DRY_RUN:-0}"

taskset=(
    "folds_tsv_DDK_ANALYSIS_PATAKA"
    "folds_tsv_SUSTAINED-VOWELS_onlyA123"
    "folds_tsv_SENTENCES"
)

case "${regime}" in
    all)
        train_file="train.tsv"
        val_file="val_early6PD6HC.tsv"
        test_file="test_early6PD6HC.tsv"
        base_seed="${BASE_SEED:-42}"
        ;;
    all_subset)
        train_file="train_allPDsubset.tsv"
        val_file="val_early6PD6HC.tsv"
        test_file="test_early6PD6HC.tsv"
        base_seed="${BASE_SEED:-42}"
        ;;
    early)
        train_file="train_earlybalance.tsv"
        val_file="val_early6PD6HC.tsv"
        test_file="test_early6PD6HC.tsv"
        base_seed="${BASE_SEED:-42}"
        ;;
    early_private)
        train_file="train_early_persp.tsv"
        val_file="val_early6PD6HC.tsv"
        test_file="test_early6PD6HC.tsv"
        base_seed="${BASE_SEED:-42}"
        taskset=(
            "folds_tsv_DDK_ANALYSIS_PATAKA"
            "folds_tsv_SUSTAINED-VOWELS_onlyA123"
        )
        ;;
    all_stage)
        train_file="train.tsv"
        val_file="val_all6PD6HC.tsv"
        test_file="test_all6PD6HC.tsv"
        base_seed="${BASE_SEED:-42}"
        ;;
    *)
        echo "Unknown regime: ${regime}" >&2
        exit 2
        ;;
esac

output_root="${OUTPUT_ROOT:-${repo_root}/outputs/reca_pd/${regime}}"

split_dir() {
    local root="$1"
    local task_dir="$2"
    local fold="$3"
    if [[ -d "${root}/${task_dir}/fold_${fold}/train_and_val" ]]; then
        printf '%s\n' "${root}/${task_dir}/fold_${fold}/train_and_val"
    elif [[ -d "${root}/${task_dir}/fold_${fold}/sub_splits" ]]; then
        printf '%s\n' "${root}/${task_dir}/fold_${fold}/sub_splits"
    else
        echo "Missing train/validation directory for ${task_dir}/fold_${fold} under ${root}" >&2
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
            train_dir="$(split_dir "${train_splits_root}" "${task_dir}" "${fold}")"
            eval_dir="$(split_dir "${splits_root}" "${task_dir}" "${fold}")"

            train_tsv="${train_dir}/${train_file}"
            val_tsv="${eval_dir}/${val_file}"
            test_tsv="${splits_root}/${task_dir}/fold_${fold}/${test_file}"
            output_dir="${output_root}/${task_dir}/run_${run_idx}/fold_${fold}"

            for required_file in "${train_tsv}" "${val_tsv}" "${test_tsv}" "${config_path}"; do
                if [[ ! -f "${required_file}" ]]; then
                    echo "Missing required file: ${required_file}" >&2
                    exit 1
                fi
            done

            command=(
                "${python_bin}" "reca_pd/scripts/model_pipeline/pipeline_new.py"
                --config "${config_path}"
                --training-dataset "${train_tsv}"
                --validation-dataset "${val_tsv}"
                --test-dataset "${test_tsv}"
                --output-dir "${output_dir}"
                --yaml-overrides "device:${device}" "seed:${seed}" "model:RECAPD"
                --save-attention-scores False
                --filter-tasks "${filter_task}"
            )

            printf '[%s] task=%s run=%s fold=%s seed=%s\n'                 "${regime}" "${filter_task}" "${run_idx}" "${fold}" "${seed}"
            if [[ "${dry_run}" == "1" ]]; then
                printf '  '
                printf '%q ' env "CUDA_VISIBLE_DEVICES=${cuda_device}" "${command[@]}"
                printf '\n'
            else
                mkdir -p "${output_dir}"
                CUDA_VISIBLE_DEVICES="${cuda_device}" "${command[@]}"
            fi
        done
    done
done
