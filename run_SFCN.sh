#!/bin/bash

python run_SFCN.py \
--data_dir "/home/nikhil/projects/brain_changes/data/ukbb/imaging/ukbb_test_subject/" \
--sfcn_ckpt "models/run_20190719_00_epoch_best_mae.p" \
--subject_list "./test_subject_list.txt" \
--scan_session "ses-2" \
--save_path "./tmp_results.csv"