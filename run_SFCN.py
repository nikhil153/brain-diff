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
                    default="/home/nikhil/projects/brain_changes/data/ukbb/",
                    help='data dir containing all the subjects')
parser.add_argument('--sfcn_ckpt', dest='sfcn_ckpt', 
                    default="models/run_20190719_00_epoch_best_mae.p",
                    help='pre-trained SFCN model weights')
parser.add_argument('--subject_list', dest='subject_list', 
                    default="sub-1010063",
                    help='subject id(s) with T1w image')
parser.add_argument('--save_path', dest='save_path', 
                    default="./tmp_results.csv",
                    help='save path for results')

args = parser.parse_args()                    

if __name__ == "__main__":
                   
    data_dir = args.data_dir
    sfcn_ckpt = args.sfcn_ckpt
    subject_list = args.subject_list
    save_path = args.save_path

    print(os.path.isfile(sfcn_ckpt))
    print(os.path.exists(data_dir))
    if (not os.path.exists(data_dir)) | (not os.path.isfile(sfcn_ckpt)):
        print(f"Either {data_dir} or {sfcn_ckpt} is missing!")
    else:
        print(f"{data_dir} and {sfcn_ckpt} found!")
        # check if it's a list of subjects
        if os.path.isfile(subject_list):
            print(f"Reading subject list from: {subject_list}")
            subject_list = list(np.hstack(pd.read_csv(subject_list,header=None).values))
            n_subjects = len(subject_list)
            print(subject_list)
            print(f"Predicting brain age for {n_subjects} subjects")
        elif subject_list in ["Random", "random"]:
            print("Generating a random scan...")
            data = np.random.rand(182, 218, 182)
        else:
            print(f"Predicting brain age for subject: {subject_list}")
            subject_list = [subject_list]

        print("Loading pre-trained model weights...")
        model = SFCN()
        model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
            
        # model.load_state_dict(torch.load(sfcn_checkpoint, map_location=torch.device('cpu')))

        results_df = pd.DataFrame(columns=["subject_id","pred","prob"])
        for s, subject_id in enumerate(subject_list):
            # Preprocessing
            #TODO

            # Prediction
            #TODO
            pred = 50
            prob = 0.9

            results = [subject_id, pred, prob]
            results_df.loc[s] = results

        print(f"Saving brain age predictions here: {save_path}")
        results_df.to_csv(save_path)
            