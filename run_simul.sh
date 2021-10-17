#!/bin/bash

RUN_ID=$1
CONFIG_ID=$2

echo "Run: $RUN_ID, CONFIG: $CONFIG_ID"
python3 run_simul.py \
--config_file "./results/simulation/configs/config_run_$RUN_ID.csv" \
--config_idx "$CONFIG_ID" \
--it "$RUN_ID" \
--save_path "/results_dir/run_$RUN_ID"
