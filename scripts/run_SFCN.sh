#!/bin/bash

pip install -e ../

python3 ../src/run_SFCN.py \
--data_dir "/home/nikhil/projects/brain_changes/data/adni/imaging/fmriprep/" \
--img_subdir "ses-bl/anat/" \
--sfcn_ckpt "../models/run_20190719_00_epoch_best_mae.p" \
--subject_list "ADNI941S5193" \
--cohort "adni" \
--scan_session "ses-bl" \
--save_path "./tmp_results.csv"
# --save_path "./brain-age_ukbb-followup_ses-2_SFCN-run-1_results.csv"
# Native T1 won't work of course