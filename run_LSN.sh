#!/bin/bash

python3 run_LSN.py \
--data_dir "/neurohub/ukbb/imaging/T1/" "/neurohub/ukbb/imaging/" \
--sfcn_ckpt "models/run_20190719_00_epoch_best_mae.p" \
--img_subdirs "ses-2/non-bids/T1/" "ses-3/anat/" \
--metadata_csv "subject_lists/metadata_train.csv" \
--batch_size "1" \
--n_epochs "1" \
--save_path "models/run_0/"
