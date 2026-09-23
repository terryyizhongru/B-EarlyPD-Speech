#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 EXPERIMENT_SETTING_DIR" >&2
  echo "Example: $0 results/reca_pd/experiments/run_5seeds_all" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
setting_dir=$1

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin=$PYTHON_BIN
elif [[ -x /data/storage1t/yzhongenv/bin/python ]]; then
  python_bin=/data/storage1t/yzhongenv/bin/python
else
  python_bin=python3
fi

"$python_bin" "$script_dir/aggregate_main_results.py" "$setting_dir"
"$python_bin" "$script_dir/aggregate_gender_results.py" "$setting_dir"
subject_args=()
if [[ -n "${SUBJECT_SAMPLE_SALT_ROOT:-}" ]]; then
  subject_args+=(--sample-salt-root "$SUBJECT_SAMPLE_SALT_ROOT")
fi
"$python_bin" "$script_dir/aggregate_subject_results.py" "$setting_dir" "${subject_args[@]}"
