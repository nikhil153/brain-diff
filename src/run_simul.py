import numpy as np
import pandas as pd
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

import matplotlib.pyplot as plt
import seaborn as sns

from src.simul import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
                    

HELPTEXT = """
Script run simulations with 2 visit data
Author: nikhil153
Date: Oct-15-2021
"""

parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--config_file', dest='config_file',  
                    default="./sim_config.csv",
                    help='path to simulation config file')

parser.add_argument('--config_idx', dest='config_idx',  
                    default=0,
                    help='config index from the dataframe')

parser.add_argument('--it', dest='it',  
                    default=0,
                    help='run iteration')

parser.add_argument('--save_path', dest='save_path', 
                    default="./",
                    help='dir path to save simulation perf')

args = parser.parse_args()

# Globals
n_timepoints = 100
n_samples_list = [100,1000]
n_regions_list = [10,100]

lr = 0.001
batch_size = 25
n_epochs = 500
hidden_node_list = [10,25,50,100]
n_jobs = 4


def run(traj_func, roi_variation, subject_variation, n_samples_list, n_regions_list, data_aug, it):
    perf_df = pd.DataFrame()
    for n_samples in n_samples_list:
        for n_regions in n_regions_list: 
            model_dict = {
                        # "Ridge": Ridge(), 
                        # "RF": RandomForestRegressor(n_jobs=n_jobs, random_state=1),                             
                        "LSN": None,                            
                        }

            traj_list = get_trajectories(traj_func, roi_variation, n_timepoints, n_regions)

            # Get roi samples
            roi_list = [get_traj_samples(traj, n_samples, subject_variation) for traj in traj_list]

            # Get cross-sectional time data (sample with replacement)
            y_baseline, X_baseline, X_followup = get_cross_sectional_samples(roi_list, followup_interval=followup_interval)

            # normalize y
            y_followup = y_baseline + followup_interval
            y_baseline = y_baseline/100
            y_followup = y_followup/100

            if followup_interval > 0:                
                X = np.hstack([X_baseline,X_followup])
                y = np.vstack([y_baseline,y_followup]).T
            else:
                X = X_baseline
                y = y_baseline
                
            # Split for CV and held-out test set
            n_CV = int(0.75 * len(y))

            X_CV = X[:n_CV]
            X_test = X[n_CV:]
            y_CV = y[:n_CV]
            y_test = y[n_CV:]

            if data_aug & (followup_interval > 0) :
                X_baseline_CV = X_baseline[:n_CV]
                X_followup_CV = X_followup[:n_CV]
                y_baseline_CV = y_baseline[:n_CV]
                y_followup_CV = y_followup[:n_CV]

                X_CV, y_CV = augment_data(X_baseline_CV, X_followup_CV, y_baseline_CV, y_followup_CV)


            # Run Models
            for model_name, model_instance in model_dict.items():
                print(f"\nSim config: iter: {it}, subject_variation: {subject_variation}, followup: {followup_interval}, \
                        n_samples: {n_samples}, n_regions= {n_regions}, model: {model_name}, \
                        traj func type: {traj_func}, shift param: {roi_variation}, data_aug: {data_aug}")

                if model_name in ["LSN","LSN1","LSN2","LSN3"]:                                            
                    trained_model_list = []
                    train_loss_list = []
                    for hidden_size in hidden_node_list:
                        if followup_interval > 0:
                            train_dataset = SimDataset(X_CV[:,:n_regions], X_CV[:,n_regions:], y_CV[:,0], y_CV[:,1])
                            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        
                            model = LSN(X_baseline.shape[1],hidden_size=hidden_size) # alternative toy model: LSN()
                            model.train()

                            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)                                                                                               
                            criterion = nn.MSELoss()                        

                            model, batch_loss_df, epoch_loss_df = train(model,train_dataloader,optimizer,criterion,n_epochs)
                        else:
                            train_dataset = SimpleSimDataset(X_CV, y_CV)
                            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        
                            model = simpleFF(X_CV.shape[1], hidden_size=hidden_size)
                            model.train()

                            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)                                                                                               
                            criterion = nn.MSELoss()                        

                            model, batch_loss_df, epoch_loss_df = trainSimpleFF(model,train_dataloader,optimizer,criterion,n_epochs)                            

                        train_loss = epoch_loss_df["epoch_loss"].values[-1]
                        train_loss_list.append(train_loss)
                        trained_model_list.append(model)
                        
                    # pick the best model
                    opt_model_idx = np.argmin(train_loss_list)
                    train_loss = np.min(train_loss_list)
                    model = trained_model_list[opt_model_idx]

                    # test
                    if followup_interval > 0:
                        test_dataset = SimDataset(X_test[:,:n_regions], X_test[:,n_regions:], y_test[:,0], y_test[:,1])
                        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                        model.eval()                    
                        batch_pred_list, test_MAE1, test_MAE2 = test(model, test_dataloader)
                        y_pred = np.vstack(np.squeeze(batch_pred_list))
                        
                        test_r1 = stats.pearsonr(y_pred[:,0],y_test[:,0])[0]
                        test_r2 = stats.pearsonr(y_pred[:,1],y_test[:,1])[0]   

                        test_age_1 = 100*y_test[:,0]
                        test_age_2 = 100*y_test[:,1]

                        test_brainage_1 = y_pred[:,0] # for two timepoints y is a matrix
                        test_brainage_2 = y_pred[:,1]

                    else:
                        test_dataset = SimpleSimDataset(X_test, y_test)
                        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

                        model.eval()                    
                        batch_pred_list, test_MAE1 = testSimpleFF(model, test_dataloader)
                        y_pred = np.squeeze(np.vstack(batch_pred_list))                                                                        
                        test_MAE2 = None
                        
                        test_r1 = stats.pearsonr(y_pred,y_test)[0]
                        test_r2 = None

                        test_age_1 = 100*y_test
                        test_age_2 = None

                        test_brainage_1 = y_pred # for single timepoint y is a vector
                        test_brainage_2 = None

                else:
                    CV_scores, y_pred, test_MAE1, test_MAE2, test_r1, test_r2 = get_brain_age_perf(X_CV, y_CV, X_test, y_test, model_instance)
                    train_loss = np.mean(-1*CV_scores) #negative MSE

                    if followup_interval > 0:
                        test_age_1 = 100*y_test[:,0]
                        test_age_2 = 100*y_test[:,1]
                        test_brainage_1 = y_pred[:,0]
                        test_brainage_2 = y_pred[:,1]
                    
                    else:
                        test_age_1 = 100*y_test
                        test_age_2 = None
                        test_brainage_1 = y_pred
                        test_brainage_2 = None
                
                df = pd.DataFrame()
                df["eid"] = np.arange(len(y_test))
                df["test_age_1"] = test_age_1
                df["test_age_2"] = test_age_2
                df["test_brainage_1"] = test_brainage_1
                df["test_brainage_2"] = test_brainage_2
                df["test_MAE1"] = test_MAE1                    
                df["test_MAE2"] = test_MAE2
                df["test_r1"] = test_r1
                df["test_r2"] = test_r2
                df["CV_score"] = train_loss
                df["model"] = model_name
                df["n_samples"] = n_samples
                df["n_regions"] = n_regions
                df["followup_interval"] = followup_interval   
                df["subject_variation"] = subject_variation
                df["traj_func"] = traj_func
                df["roi_variation"] = roi_variation
                df["data_aug"] = data_aug
                df["iter"] = it

                perf_df = perf_df.append(df)

    return perf_df


if __name__ == "__main__":
    # Read from csv
    config_file = args.config_file
    config_idx = int(args.config_idx)
    it = args.it
    save_path = f"{args.save_path}/sim_perf_config_{config_idx}.csv"

    config_df = pd.read_csv(config_file)

    traj_func = config_df.loc[config_idx,"traj_func"]
    roi_variation = config_df.loc[config_idx,"roi_variation"]
    subject_variation = config_df.loc[config_idx,"subject_variation"]
    followup_interval = config_df.loc[config_idx,"followup_interval"]
    data_aug = config_df.loc[config_idx,"data_aug"]

    perf_df = run(traj_func, roi_variation, subject_variation, n_samples_list, n_regions_list, data_aug, it)

    print(f"Saving simulation run:{it}, config: {config_idx} results at: {save_path}")
    perf_df.to_csv(save_path)
