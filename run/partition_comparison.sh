#!/bin/bash
set -eou pipefail

python3 main.py \
	--yaml-cfg run/partition_cfg.yaml \
	--workload-cfg run/workloads.yaml
