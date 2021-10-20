#!/bin/bash

pip install -e ../

python3 run_SFCN.py \
--data_dir "/neurohub/ukbb/imaging/" \
--img_subdir "ses-3/anat/" \
--sfcn_ckpt "../models/run_0/lsn.ckpt" \
--subject_list "../metadata/ukbb_brain-age_follow-up_subjects.txt" \
--scan_session "ses-3" \
--save_path "../results/brain-age_ukbb-followup_ses-3_LSN-run-1_results.csv"
