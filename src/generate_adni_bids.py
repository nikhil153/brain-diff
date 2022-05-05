import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import shutil
from pathlib import Path

HELPTEXT = """
Script to generate bids formatted adni T1w data
Author: nikhil153
Date: Apr-25-2022
"""

# Sample cmd:
#  python generate_adni_bids.py --nii_path_file ~/projects/brain_changes/brain-diff/metadata/adni/ADNI123_nii_paths.txt \
#                               --bids_dir ~/projects/brain_changes/brain-diff/test_data/bids/  \
#                               --adnimerge_file ~/projects/brain_changes/brain-diff/metadata/adni/ADNIMERGE.csv \
#                               --log_dir ./ \
#                               --test_run

parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--nii_path_file', dest='nii_path_file',                     
                    help='path to file with all the nii (abslolute) paths extrated from ADNI longitudinal zip files')

parser.add_argument('--bids_dir', dest='bids_dir',  
                    default='./bids_dir',
                    help='path to bids_dir')

parser.add_argument('--adnimerge_file', dest='adnimerge_file',  
                    default='./adnimerge_file',
                    help='path to adnimerge_file')

parser.add_argument('--test_run',  
                    action='store_true',
                    help='do a test run with 5 images')

parser.add_argument('--log_dir', dest='log_dir',  
                    default='./log_dir',
                    help='path to log_dir')

args = parser.parse_args()


def get_subject_info_from_path(p):
    nii_file_name = p.rsplit("/",1)[1]
    ptid = nii_file_name.split("_")[1] + "_S_" + nii_file_name.split("_")[3]
    acq_date = p.rsplit("/",3)[1].split("_")[0]
    img_id = nii_file_name[::-1].split(".")[1].split("_")[0][::-1]

    return ptid, acq_date, img_id

def get_closest_visit_code(subject_df, acq_date):
    # find the closest date match
    date_diff_list = (pd.to_datetime(subject_df['EXAMDATE'].values, yearfirst=True, format="%Y/%m/%d") - 
                pd.to_datetime(acq_date, yearfirst=True, format="%Y/%m/%d"))

    closest_date_idx = np.argmin(np.abs(date_diff_list))
    diff_in_days = np.min(np.abs(date_diff_list)).days
    closest_date = subject_df['EXAMDATE'].values[closest_date_idx]

    visit_code = subject_df[subject_df["EXAMDATE"]==closest_date]['VISCODE'].values[0]

    return visit_code, diff_in_days


if __name__ == "__main__":
    # Read from csv
    nii_path_file = args.nii_path_file
    bids_dir = args.bids_dir
    log_dir = args.log_dir
    adnimerge_file = args.adnimerge_file
    test_run = args.test_run

    nii_path_list = list(pd.read_csv(nii_path_file,header=None)[0].values)
    print(f"number of nii paths found: {len(nii_path_list)}")

    if test_run:
        path_list = nii_path_list[:5]
        print(f"Doing a test run with 5 images")
    else: 
        path_list = nii_path_list

    adnimerge_df = pd.read_csv(adnimerge_file)

    log_df = pd.DataFrame()
    for p in path_list:
        ptid, acq_date, img_id = get_subject_info_from_path(p)

        subject_df = adnimerge_df[(adnimerge_df["PTID"]==ptid)]
        
        visit_code, diff_in_days = get_closest_visit_code(subject_df, acq_date)

        sub_label = "sub-ADNI" + ptid.replace("_","")
        ses_label = f"ses-{visit_code}"
        
        save_dir = f"{bids_dir}/{sub_label}/{ses_label}/anat/"
        bids_name = sub_label + "_" + ses_label + "_T1w.nii"
        save_path = f"{save_dir}{bids_name}"
        
        df = pd.DataFrame(columns=["PTID","IID","visit_code","acq_date","diff_in_days","bids_name"])
        df.loc[0] = [ptid, img_id, visit_code, acq_date, diff_in_days, bids_name]
        
        log_df = pd.concat([log_df,df],axis=0)

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, save_path)

    log_df.to_csv(f"{log_dir}/bids.log")

    print(f"Saving log at : {log_dir}")

