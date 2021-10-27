#!/bin/bash

RUN_ID=$1
CONFIG_ID=$2
DATA_DIR=$3
RESULTS_DIR=$4

pip install -e ../

echo "RUN: $RUN_ID, CONFIG: $CONFIG_ID, DATA DIR: $DATA_DIR, RESULTS DIR: $RESULTS_DIR"

python3 ../src/run_LSN_roi.py \
--data_dir "$DATA_DIR/" \
--metadata_dir "../metadata/" \
--config_file "../results/LSN_roi/configs/config_run_$RUN_ID.csv" \
--config_idx "$CONFIG_ID" \
--run_id "$RUN_ID" \
--save_path "$RESULTS_DIR/run_$RUN_ID/" \
--mock_run 0
