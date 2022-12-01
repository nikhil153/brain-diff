import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import glob
from bids import BIDSLayout 
import os


HELPTEXT = """
Script to check participant-session availability 
"""
#Author: nikhil153
#Date: 1-Dec-2022

# Globals
F_EXT = 'nii.gz' 
F_SUFFIX = "T1w"

# argparse
parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--bids_dir', help='path to bids_dir with all the subjects')
parser.add_argument('--output_csv', help='path to output csv file')                    

args = parser.parse_args()
bids_dir = args.bids_dir
output_csv = args.output_csv
participants_tsv = f"{bids_dir}/participants.tsv"


# Check participants tsv and actual participant dirs
tsv_participants = set(pd.read_csv(participants_tsv,sep="\t")["participant_id"].values)
bids_dir_paths = glob.glob(f"{bids_dir}/sub*")
bids_dir_participants = set([os.path.basename(x) for x in bids_dir_paths])

participants_missing_in_tsv = bids_dir_participants - tsv_participants
participants_missing_in_bids_dir = tsv_participants - bids_dir_participants

print(f"n_participants_tsv: {len(tsv_participants)}, \
        n_participants_bids_dir: {len(bids_dir_participants)}, \
        n_participants_missing_in_tsv: {len(participants_missing_in_tsv)}, \
        n_participants_missing_in_bids_dir: {len(participants_missing_in_bids_dir)}")


if tsv_participants == bids_dir_participants:
    layout = BIDSLayout(bids_dir)
    session_list = layout.get_sessions()

    bids_status_df = pd.DataFrame(index=tsv_participants, columns=session_list)    

    for participant in tsv_participants:
        participant_id = participant.split("-",2)[1]
        for ses in session_list:
            f = layout.get(subject=participant_id, 
                                    session=ses, 
                                    extension=F_EXT, 
                                    suffix=F_SUFFIX,                 
                                    return_type='filename')
            if f != None:
                bids_status_df.loc[participant,ses] = len(f)

    print(f"Saving bids_status_df at {output_csv}")        
    bids_status_df.to_csv(output_csv)

else:
    print(f"participants_tsv and bids_dir participants mismatch...")


