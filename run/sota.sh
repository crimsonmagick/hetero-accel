#!/bin/bash
set -eou pipefail

node="$(tmux display -p '#S')"

python3 main.py \
	--yaml-cfg run/sota_cfg.yaml \
	--workload-cfg run/workloads.yaml

final_results="$(grep -A 4 "Evaluation results:" latest_log_file | tail -n 5)"
curl -d "[$node] sota completed: $final_results" ntfy.sh/hetero-accel
