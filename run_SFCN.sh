#!/bin/bash

python3 run_SFCN.py \
--data_dir "/neurohub/ukbb/imaging/T1/" \
--sfcn_ckpt "models/run_20190719_00_epoch_best_mae.p" \
--subject_list "./test_subject_list.txt" \
--scan_session "ses-2" \
--save_path "./tmp_results.csv"
# --save_path "./brain-age_ukbb-followup_ses-2_SFCN-run-1_results.csv"

