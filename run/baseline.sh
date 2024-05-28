#!/bin/bash
set -eou pipefail

python3 main.py \
	--yaml-cfg run/baseline_cfg.yaml \
	--workload-cfg run/workloads.yaml


final_results="$(grep -A 4 "Evaluation results:" logs/ours___2024.05.28-15.48.41.131/ours___2024.05.28-15.48.41.131.log | tail -n 5)"
curl -d "baseline completed: $final_results" ntfy.sh/hetero-accel
