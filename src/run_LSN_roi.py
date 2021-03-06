import numpy as np
import pandas as pd
import time
from datetime import datetime
import argparse

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.utils import *
from src.LSN_roi import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
                    

HELPTEXT = """
Script to run ukbb brainage with 2 visit freesurfer data
Author: nikhil153
Date: Oct-27-2021
"""

# Sample run command
# python3 run_LSN_roi.py --config_file ../results/LSN_roi/configs/config_run_3.csv --config_idx 0 --run_id 3 \
# --data_dir ../../data/ukbb/imaging/freesurfer/ --metadata_dir ../metadata/ \
# --save_path ../results/LSN_roi/run_3/

parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--config_file', dest='config_file',  
                    default="./config.csv",
                    help='path to simulation config file')

parser.add_argument('--config_idx', dest='config_idx',  
                    default=0,
                    help='config index from the dataframe')

parser.add_argument('--run_id', dest='run_id',  
                    default=0,
                    help='run iteration')

parser.add_argument('--data_dir', dest='data_dir',  
                    default="../../data/",
                    help='path to freesurfer data dir')

parser.add_argument('--metadata_dir', dest='metadata_dir',  
                    default="../metadata/",
                    help='path to metadata dir')

parser.add_argument('--save_path', dest='save_path', 
                    default="../results/LSN_roi/run_0/",
                    help='dir path to save simulation perf')

parser.add_argument('--mock_run', dest='mock_run', 
                    default=0,
                    help='flag for mock run with small subset')

args = parser.parse_args()

# Globals
lr = 0.001
batch_size = 5
n_epochs = 50

def run(train_df, test_df, pheno_cols_ses2, pheno_cols_ses3, hidden_size, transform):
    # train
    train_dataset = UKBB_ROI_Dataset(train_df, pheno_cols_ses2, pheno_cols_ses3, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_size = len(pheno_cols_ses2)
    model = LSN_FF_Linear(input_size,hidden_size=hidden_size) # alternative toy model: LSN()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optim.SGD(model.parameters(), lr=lr, momentum=0.5)                                                                                               
    criterion = nn.MSELoss()                        

    # using subset of train dataloader for debug
    model, batch_loss_df, epoch_loss_df, preds_df = train(model,train_dataloader,optimizer,criterion,n_epochs)

    # test
    perf_df = pd.DataFrame()

    for test_transform in [None, "swap"]:
        if test_transform == "swap":
            visit_order = "F,B"
            y_test = test_df[["age_at_ses3", "age_at_ses2"]].values 
        else:
            visit_order = "B,F"
            y_test = test_df[["age_at_ses2", "age_at_ses3"]].values 

        test_dataset = UKBB_ROI_Dataset(test_df, pheno_cols_ses2, pheno_cols_ses3, transform=test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model.eval()

        eid_list, y_test_list, y_pred_list, loss1_list, loss2_list = test(model, test_dataloader)

        y_test = np.squeeze(np.vstack(y_test_list))
        y_pred = np.squeeze(np.vstack(y_pred_list))

        test_r1 = stats.pearsonr(y_pred[:,0],y_test[:,0])[0]
        test_r2 = stats.pearsonr(y_pred[:,1],y_test[:,1])[0]                                     

        df = pd.DataFrame()
        df["eid"] = test_df["eid"]
        df["age_at_ses2"] = y_test[:,0]
        df["age_at_ses3"] = y_test[:,1]
        df["brainage_at_ses2"] = y_pred[:,0]
        df["brainage_at_ses3"] = y_pred[:,1]
        df["test_loss1"] = loss1_list                    
        df["test_loss2"] = loss2_list
        df["test_r1"] = test_r1
        df["test_r2"] = test_r2
        df["visit_order"] = visit_order
        
        perf_df = perf_df.append(df)

    return epoch_loss_df, perf_df


if __name__ == "__main__":
    # Read from csv
    config_file = args.config_file
    config_idx = int(args.config_idx)
    run_id = args.run_id
    perf_save_path = f"{args.save_path}/freesurfer_perf_config_{config_idx}.csv"
    train_save_path = f"{args.save_path}/freesurfer_train_loss_config_{config_idx}.csv"
    mock_run = int(args.mock_run)

    config_df = pd.read_csv(config_file)
    hidden_size = config_df.loc[config_idx,"hidden_size"]
    transform = config_df.loc[config_idx,"transform"]
    phenotype = config_df.loc[config_idx,"phenotype"]

    print(f"Config: hidden_size: {hidden_size}, transform: {transform}, pheno: {phenotype}")

    metadata_dir = args.metadata_dir #"../"
    data_dir = args.data_dir #"/home/nikhil/projects/brain_changes/data/ukbb/"

    print(f"data dir: {data_dir}, metadata_dir: {metadata_dir}")

    train_csv = f"{metadata_dir}/metadata_train.csv"
    test_csv = f"{metadata_dir}/metadata_test.csv"

    ## Select freesurfer phenotype (e.g. thickness vs volume vs both)
    freesurfer_fields = f"{metadata_dir}/ukbb_freesurfer_fields.txt"
    freesurfer_fields_df = pd.read_csv(freesurfer_fields,sep="	")
    freesurfer_fields_df["phenotype"] = freesurfer_fields_df["Description"].str.split(" ",1,expand=True)[0]
    freesurfer_fields_df["phenotype"] = freesurfer_fields_df["phenotype"].replace({"Mean":"Mean Thickness"})
    CT_fields = freesurfer_fields_df[freesurfer_fields_df["phenotype"]=="Mean Thickness"]["Field ID"]
    volume_fields = freesurfer_fields_df[freesurfer_fields_df["phenotype"]=="Volume"]["Field ID"]

    if phenotype in ["CT", "cortical thickness"]:
        pheno_fields = CT_fields
    elif phenotype == "volume":
        pheno_fields = volume_fields
    else:
        pheno_fields = CT_fields.append(volume_fields)

    # print(f"phono fields: {pheno_fields}")
    pheno_cols_ses2 = list(pheno_fields.astype(str) + "-2.0")
    pheno_cols_ses3 = list(pheno_fields.astype(str) + "-3.0")
    usecols = ["eid"] + pheno_cols_ses2 + pheno_cols_ses3

    freesurfer_csv = f"{data_dir}ukb47552_followup_subset.csv"
    data_df = pd.read_csv(freesurfer_csv, usecols=usecols)
    

    train_df, test_df = get_ML_dataframes(usecols, freesurfer_csv, train_csv, test_csv)
    if mock_run == 1:
        print(f"Doing a mock run with 100 train samples and 10 test samples")
        train_df = train_df.head(100)
        test_df = test_df.head(10)

    start_time = time.time()

    epoch_loss_df, perf_df = run(train_df, test_df, pheno_cols_ses2, pheno_cols_ses3, hidden_size, transform)

    print(f"Saving LSN_roi run:{run_id}, config: {config_idx} results at: {perf_save_path}")
    perf_df.to_csv(perf_save_path)
    epoch_loss_df.to_csv(train_save_path)

    end_time = time.time()
    run_time = (end_time - start_time)/3600.0

    print(f"total run time (hrs): {run_time}") 
