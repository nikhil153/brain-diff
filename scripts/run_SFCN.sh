#!/bin/bash

pip install -e ../

python3 ../src/run_SFCN.py \
--data_dir "/home/nikhil/scratch/adni_processing/fmriprep/ohbm/followup/output/fmriprep/" \
--img_subdir "ses-m24/anat/" \
--sfcn_ckpt "../models/run_20190719_00_epoch_best_mae.p" \
--subject_list "/home/nikhil/scratch/my_repos/brain-diff/metadata/adni_long_ohbm_subject_ids.txt" \
--cohort "adni" \
--apply_brain_mask \
--scan_session "ses-m24" \
--save_path "./adni_sfcn_ohbm_m24_masked_results.csv"
# --save_path "./brain-age_ukbb-followup_ses-2_SFCN-run-1_results.csv"
# Native T1 won't work of course
