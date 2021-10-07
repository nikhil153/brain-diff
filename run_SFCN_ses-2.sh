#!/bin/bash

python3 run_SFCN.py \
--data_dir "/neurohub/ukbb/imaging/T1/" \
--img_subdir "ses-2/non-bids/T1/" \
--sfcn_ckpt "models/run_0/lsn.ckpt" \
--subject_list "subject_lists/ukbb_brain-age_follow-up_subjects.txt" \
--scan_session "ses-2" \
--save_path "./brain-age_ukbb-followup_ses-2_LSN-run-1_results.csv"
