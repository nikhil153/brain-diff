from pathlib import Path
import argparse
import glob
import os
import pandas as pd
import tarfile


HELPTEXT = """
Script to tar set of files from fmriprep output dir (reduce i-nodes) 
"""
#Author: nikhil153
#Date: 19-Dec-2022

# helper function
def tar_files(tar_file_name, file_list):
    arcname = os.path.basename(tar_file_name).rsplit(".",1)[0]
    tar_file= tarfile.open(tar_file_name,"w")
    for f in file_list:
        f_basename = os.path.basename(f)
        tar_file.add(f,arcname=f"{arcname}/{f_basename}")
    tar_file.close()  

# argparse
parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--fmriprep_dir', help='path to bids_dir with all the subjects')
parser.add_argument('--session', help='session id')
parser.add_argument('--modality', default="anat", help='modality')
parser.add_argument('--participants_list', help='path to list of particitpants')     
parser.add_argument('--file_ext', default="", help='file extension')               
parser.add_argument('--output_dir', default=None, help='output_dir to save tars')
parser.add_argument('--remove_orig', default=False, action='store_true', help='remove original files after tarring to clean up')


args = parser.parse_args()


fmriprep_dir = args.fmriprep_dir
session = args.session
modality = args.modality
participants_list = args.participants_list
file_ext = args.file_ext
output_dir = args.output_dir
remove_orig = args.remove_orig

if file_ext == "":
    print("No file-type provided. Tarring entire participant dir")

if output_dir == None:
    print("Saving tars in to the participant dir itself")
    
if remove_orig:
    print("***Removing original files after tarring***")

# Check number of participants from the list
if participants_list.rsplit(".")[1] == "tsv":
    participants_df = pd.read_csv(participants_list,sep="\t")
else:
    participants_df = pd.read_csv(participants_list)

participant_ids = participants_df["participant_id"]
if str(participant_ids.values[0])[:3] != "sub":
    print("Adding sub prefix to the participant_id(s)")
    participant_ids = ["sub-" + str(id) for id in participant_ids]
    
print(f"Number of participants: {len(participant_ids)}")
for participant_id in participant_ids:
    participant_dir = f"{fmriprep_dir}/{participant_id}/ses-{session}/{modality}/"
    file_list = glob.glob(f"{participant_dir}/*{file_ext}")

    if output_dir == None:
        save_dir = participant_dir
    else:
        save_dir = output_dir

    if file_ext == "":
        tar_file_name = f"{save_dir}/{participant_id}.tar"
    else:
        tar_file_name = f"{save_dir}/{participant_id}_{file_ext}.tar"

    # Tar the files
    try:
        tar_files(tar_file_name, file_list)
    except:
        print(f"Error while tarring {participant_id}")
    else:
        if remove_orig:
            for f in file_list:
                os.remove(f)