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
parser.add_argument('--sfcn_ckpt', dest='sfcn_ckpt', 
                    default="models/run_20190719_00_epoch_best_mae.p",
                    help='pre-trained SFCN model weights')
parser.add_argument('--subject_list', dest='subject_list', 
                    default="1010063",
                    help='subject id(s) with T1w image')
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
    sfcn_ckpt = args.sfcn_ckpt
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
            
        model.load_state_dict(torch.load(sfcn_ckpt, map_location=torch.device('cpu')))

        results_df = pd.DataFrame(columns=["eid","pred","prob"])
        for s, subject_id in enumerate(subject_list):
            try:
                # Load image
                subject_dir = f"{data_dir}sub-{subject_id}/{scan_session}/non-bids/T1/"
                T1_mni = f"{subject_dir}T1_brain_to_MNI.nii.gz"
                data = nib.load(T1_mni).get_fdata()

                # Preprocessing
                input_data = preproc_images(data)

                # Prediction
                prob, pred = get_brain_age(input_data, model, bc)
            
                results = [subject_id, pred, prob]
                results_df.loc[s] = results
            
            except:
                print(f"Could not read T1w data for :{subject_id}")                
                continue
        
        # Save results to a csv
        print(f"Saving brain age predictions here: {save_path}")
        results_df.to_csv(save_path)      
