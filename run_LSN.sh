#!/bin/bash

python3 run_LSN.py \
--data_dir "/home/nikhil/projects/brain_changes/data/ukbb/imaging/ukbb_test_subject/" \
--sfcn_ckpt "models/lsn.ckpt" \
--img_subdirs "ses-2/non-bids/T1/" "ses-2/non-bids/T1/" \
--metadata_csv "/home/nikhil/projects/brain_changes/data/ukbb/tabular/ukbb_test_subject_metadata.csv" \
--batch_size "2" \
--n_epochs "2" \
--save_path "models/run_0/"