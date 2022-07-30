from re import ASCII
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


HELPTEXT = """
Script to validate freesurfer output
"""
#Author: nikhil153
#Date: 27-July-2022


# Sample cmd:


# Globals
FIRST_LEVEL_DIRS = ["label", "mri", "stats", "surf"]
HEMISPHERES = ["lh","rh"]
PARCELS = ["aparc","aparc.a2009s","aparc.DKTatlas"]
SURF_MEASURES = ["curv","area","thickness","volume","sulc","midthickness"]

# argparse
parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--fs_output_dir', dest='fs_output_dir',                      
                    help='path to fs_output_dir with all the subjects')

parser.add_argument('--participants_list', dest='participants_list',                      
                    help='path to participants list (csv or tsv')

args = parser.parse_args()

def check_fsdirs(subject_dir):
    status_msg = "Pass"
    for fsdir in FIRST_LEVEL_DIRS:
        dirpath = Path(f"{subject_dir}/{fsdir}")
        dirpath_status = Path.is_dir(dirpath)
        if not dirpath_status:
            status_msg = f"{fsdir} not found"
            break;
    return status_msg

def check_mri(mri_dir):
    status_msg = "Pass"
    for parc in PARCELS:
        filepath = Path(f"{mri_dir}/{parc}+aseg.mgz")
        filepath_status = Path.is_file(filepath)
        if not filepath_status:
            status_msg = f"{parc}+aseg.mgz not found"
            break;
    return status_msg

def check_label(label_dir):
    status_msg = "Pass"
    for parc in PARCELS:
        for hemi in HEMISPHERES:
            filepath = Path(f"{label_dir}/{hemi}.{parc}.annot")
            filepath_status = Path.is_file(filepath)
            if not filepath_status:
                status_msg = f"{hemi}.{parc}.annot not found"
                break;
    return status_msg

def check_stats(stats_dir):
    status_msg = "Pass"
    for parc in PARCELS:
        for hemi in HEMISPHERES:
            filepath = Path(f"{stats_dir}/{hemi}.{parc}.stats")
            filepath_status = Path.is_file(filepath)
            if not filepath_status:
                status_msg = f"{hemi}.{parc}.stats not found"
                break;

    # check aseg
    filepath = Path(f"{stats_dir}/aseg.stats")
    filepath_status = Path.is_file(filepath)
    if not filepath_status:
        status_msg = f"{parc}.stats not found"

    return status_msg

def check_surf(surf_dir):
    status_msg = "Pass"
    for measure in SURF_MEASURES:
        for hemi in HEMISPHERES:
            filepath = Path(f"{surf_dir}/{hemi}.{measure}")
            filepath_status = Path.is_file(filepath)
            if not filepath_status:
                status_msg = f"{hemi}.{measure} not found"
                break;
    return status_msg



def check_output(subject_dir):

    # check fsdirs
    fsdir_status = check_fsdirs(subject_dir)
    
    # check mris
    mri_dir = f"{subject_dir}/mri/"
    mri_status = check_mri(mri_dir)

    # check labels
    label_dir = f"{subject_dir}/label/"
    label_status = check_label(label_dir)

    # check stats
    stats_dir = f"{subject_dir}/stats/"
    stats_status = check_stats(stats_dir)

    # check surf
    surf_dir = f"{subject_dir}/surf/"
    surf_status = check_surf(surf_dir)

    return [fsdir_status, mri_status, label_status, stats_status, surf_status]

if __name__ == "__main__":
    # Read from csv
    fs_output_dir = args.fs_output_dir
    participants_list = args.participants_list

    if participants_list.rsplit(".")[1] == "tsv":
        participants_df = pd.read_csv(participants_list,sep="\t")
    else:
        participants_df = pd.read_csv(participants_list)

    participant_ids = participants_df["participant_id"]
    n_participans = len(participant_ids)
    print(f"Checking FreeSurfer output for {n_participans} subjects")

    status_df = pd.DataFrame(columns=["participant_id","fsdir_status","mri_status","label_status","stats_status","surf_status"])
    for p, participant_id in enumerate(participant_ids):
        subject_dir = f"{fs_output_dir}/{participant_id}"
        status_list = check_output(subject_dir)
        status_df.loc[p] = [participant_id] + status_list
        
    # Save fs_status_df
    save_path = "./fs_status.csv"
    print(f"Status log location: {save_path}")
    status_df.to_csv(save_path)