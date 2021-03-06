import numpy as np
import pandas as pd
import seaborn as sns
import os 

from models.sfcn import *
from models import dp_loss as dpl
from models import dp_utils as dpu
import nibabel as nib

import argparse

import torch
import torch.nn.functional as F


HELPTEXT = """
Script to predict brain age using SFCN model and list of T1w scans
Author: nikhil153
Date: Sept-17-2021
"""

parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--data_dir', dest='data_dir', 
                    default="/home/nikhil/projects/brain_changes/data/ukbb/imaging/ukbb_test_subject/",
                    help='data dir containing all the subjects')
parser.add_argument('--img_subdir', dest='img_subdir',
                    default="ses-2/non-bids/T1/",
                    help='path from subject dir to T1 scan')
parser.add_argument('--sfcn_ckpt', dest='sfcn_ckpt', 
                    default="models/run_20190719_00_epoch_best_mae.p",
                    help='pre-trained SFCN model weights')
parser.add_argument('--cohort', dest='cohort', 
                    default="ukbb",
                    help='subject cohort e.g. ukbb, adni')
parser.add_argument('--subject_list', dest='subject_list', 
                    default="1010063",
                    help='subject id(s) with T1w image')
parser.add_argument('--apply_brain_mask', action='store_true', 
                    help='flag for applying brainmask to T1w')
parser.add_argument('--scan_session', dest='scan_session', 
                    default="ses-2",
                    help='scan session for the T1w image')
parser.add_argument('--save_path', dest='save_path', 
                    default="./tmp_results.csv",
                    help='save path for results')

args = parser.parse_args()

def get_brain_age(input_data, model, bc):
    """ Function to get brain age from T1w MRI (linear reg to MNI space) and SFCN model checkpoint
    """
    model.eval() 
    with torch.no_grad():
        output = model.module(input_data)

    # Output, loss, visualisation
    x = output[0].reshape([1, -1])

    x = x.numpy().reshape(-1)
    prob = np.exp(x)
    pred = prob@bc

    return prob, pred

def preproc_images(img, crop_shape=(160, 192, 160)):
    """ Function to preprocess T1w scan as expected by SFCN
    """
    img = img/img.mean()
    img = dpu.crop_center(img, crop_shape)

    # Move the img from numpy to torch tensor
    sp = (1,1)+img.shape
    img = img.reshape(sp)
    input_data = torch.tensor(img, dtype=torch.float32)

    return input_data

if __name__ == "__main__":
                   
    data_dir = args.data_dir
    img_subdir = args.img_subdir
    sfcn_ckpt = args.sfcn_ckpt
    cohort = args.cohort
    apply_brain_mask = args.apply_brain_mask
    subject_list = args.subject_list
    scan_session = args.scan_session 
    
    # Changing this range will shift the predicted age.
    # Prediction is treated as classification problem with n_classes = n_bins
    age_range = [42,82]
    bin_step = 1
    bc = dpu.get_bin_centers(age_range, bin_step)

    save_path = args.save_path

    if (not os.path.exists(data_dir)) | (not os.path.isfile(sfcn_ckpt)):
        print(f"Either {data_dir} or {sfcn_ckpt} is missing!")
    else:
        print(f"{data_dir} and {sfcn_ckpt} found!")
        # check if it's a list of subjects
        if os.path.isfile(subject_list):
            print(f"Reading subject list from: {subject_list}")
            subject_list = list(np.hstack(pd.read_csv(subject_list,header=None).values))
            n_subjects = len(subject_list)
            print(f"Predicting brain age for {n_subjects} subjects")
        elif subject_list in ["Random", "random"]:
            print("Generating a random scan...")
            data = np.random.rand(182, 218, 182)
        else:
            print(f"Predicting brain age for subject: {subject_list}")
            subject_list = [subject_list]

        # Load model
        print("Loading pre-trained model weights...")
        model = SFCN()
        model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'


        # Original SFCN model
        if sfcn_ckpt.rsplit("/",1)[1] == "run_20190719_00_epoch_best_mae.p":
            print("Using original SFCN checkpoint")
            model.load_state_dict(torch.load(sfcn_ckpt, map_location=torch.device('cpu')))
        else:
            print("Using fine-tuned LSN checkpoint")
            checkpoint = torch.load(sfcn_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])

        results_df = pd.DataFrame(columns=["eid","pred","prob"])
        for s, subject_id in enumerate(subject_list):
            try:
                # Load image
                subject_dir = f"{data_dir}sub-{subject_id}/"
                if cohort == "adni":
                    T1_filename = f"sub-{subject_id}_{scan_session}_space-MNI152Lin_res-1_desc-preproc_T1w.nii.gz"
                    brainmask_filename = f"sub-{subject_id}_{scan_session}_space-MNI152Lin_res-1_desc-brain_mask.nii.gz"
                    
                elif cohort == "ukbb":
                    if scan_session == "ses-2":
                        T1_filename = "T1_brain_to_MNI.nii.gz"
                    elif scan_session == "ses-3":
                        T1_filename = f"{subject_id}_ses-3_T1_brain_to_MNI.nii.gz"
                    else:
                        print(f"Unknown scan session: {scan_session} for {cohort} cohort")

                else:
                    print(f"Unknown {cohort} cohort")


                T1_mni = f"{subject_dir}{img_subdir}{T1_filename}"
                print(f"T1 path: {T1_mni}")
                T1_data = nib.load(T1_mni).get_fdata()

                # Apply brain mask
                if apply_brain_mask:
                    print("applying brain mask")
                    brainmask_mni = f"{subject_dir}{img_subdir}{brainmask_filename}"
                    brainmask_data = nib.load(brainmask_mni).get_fdata()
                    masked_T1_data = brainmask_data * T1_data
                    # Preprocessing
                    input_data = preproc_images(masked_T1_data)

                else:
                    # Preprocessing
                    input_data = preproc_images(T1_data)

                # Prediction
                prob, pred = get_brain_age(input_data, model, bc)
            
                results = [subject_id, pred, prob]
                results_df.loc[s] = results
            
            except Exception as e: # work on python 3.x
                print(f"Could not read T1w data for: {subject_id} because {e}")                
                continue
        
        # Save results to a csv
        print(f"Saving brain age predictions here: {save_path}")
        results_df.to_csv(save_path)      
