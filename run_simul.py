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

from simul import *

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
batch_size = 25
n_epochs = 100
hidden_node_list = [10,100]
n_jobs = 4

def get_trajectories(traj_func, roi_variation, n_timepoints, n_regions):
    """ Generates entire trajectories for a specified trajectory function and roi_variation parameters
    """
    # Get traj(s): subject values are shifted in intercept
    traj_list = []
    if traj_func == "exp":
        init_val = 10 # same as peak val i.e. max thickness
        init_val_min = 1
        init_val_max = 10         
        decay_min = 10
        decay_max = 80
        decay = 50

        # Region values decay in time
        decay_list = np.linspace(start=decay_min, stop=decay_max, num=n_regions)
        init_val_list = np.linspace(start=init_val_min, stop=init_val_max, num=n_regions)

        if roi_variation == "roi_time":
            for decay in decay_list:
                func_params = {"init_val": init_val, "decay": decay}
                traj_list.append(get_brain_trajectory(n_timepoints, traj_func, func_params))
        elif roi_variation == "roi_init":
            for init_val in init_val_list:
                func_params = {"init_val": init_val, "decay": decay}
                traj_list.append(get_brain_trajectory(n_timepoints, traj_func, func_params))
        else:
            for init_val, decay in zip(init_val_list, decay_list):
                func_params = {"init_val": init_val, "decay": decay}
                traj_list.append(get_brain_trajectory(n_timepoints, traj_func, func_params))

    elif traj_func == "poly":
        init_val = 1 
        init_val_min = 1
        init_val_max = 9

        peak_val = 10 #max thickness 
        
        roi_maturity = 65
        roi_maturity_min = 50
        roi_maturity_max = 80

        # Region values are shifted in time
        roi_maturity_list = np.linspace(start=roi_maturity_min,stop=roi_maturity_max,num=n_regions)
        init_val_list = np.linspace(start=init_val_min,stop=init_val_max,num=n_regions)

        if roi_variation == "roi_time":
            for roi_maturity in roi_maturity_list:
                func_params = {"init_val": init_val, "peak_val": peak_val, "time_shift": roi_maturity, "poly_order": 2}
                traj_list.append(get_brain_trajectory(n_timepoints, traj_func, func_params))

        elif roi_variation == "roi_init":
            for init_val in init_val_list:
                func_params = {"init_val": init_val, "peak_val": peak_val, "time_shift": roi_maturity, "poly_order": 2}
                traj_list.append(get_brain_trajectory(n_timepoints, traj_func, func_params))

        else:
            for time_shift, init_val in zip(roi_maturity_list, init_val_list):
                func_params = {"init_val": init_val, "peak_val": peak_val, "time_shift": time_shift, "poly_order": 2}
                traj_list.append(get_brain_trajectory(n_timepoints, traj_func, func_params))

    else:
        print(f"Unknown func type: {traj_func}")

    return traj_list

def augment_data(X_baseline_CV, X_followup_CV, y_baseline_CV, y_followup_CV, swap_only=True):
    """ Augments training (i.e. internal CV data) by swapping baseline and followup data
    """
    X_orig = np.hstack([X_baseline_CV,X_followup_CV])
    y_orig = np.vstack([y_baseline_CV,y_followup_CV]).T
    
    # swap timepoints
    X_swap = np.hstack([X_followup_CV, X_baseline_CV])
    y_swap = np.vstack([y_followup_CV, y_baseline_CV]).T
    
    # baseline only
    X_base = np.hstack([X_baseline_CV, X_baseline_CV])
    y_base = np.vstack([y_baseline_CV, y_baseline_CV]).T

    # followup only
    X_follow = np.hstack([X_followup_CV, X_followup_CV])
    y_follow = np.vstack([y_followup_CV, y_followup_CV]).T

    if swap_only:
        X_CV = np.vstack([X_orig,X_swap])
        y_CV = np.vstack([y_orig,y_swap])
    else:
        X_CV = np.vstack([X_orig,X_swap,X_base,X_follow])
        y_CV = np.vstack([y_orig,y_swap,y_base,y_follow])
    
    return X_CV, y_CV


def run(traj_func, roi_variation, subject_variation, n_samples_list, n_regions_list, data_aug, it):
    perf_df = pd.DataFrame()
    for n_samples in n_samples_list:
        for n_regions in n_regions_list: 
            model_dict = {
                        "Ridge": Ridge(), 
                        "RF": RandomForestRegressor(n_jobs=n_jobs, random_state=1),                             
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

            if data_aug:
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

                            optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)                                                                                               
                            criterion = nn.MSELoss()                        

                            model, batch_loss_df, epoch_loss_df = train(model,train_dataloader,optimizer,criterion,n_epochs)
                        else:
                            train_dataset = SimpleSimDataset(X_CV, y_CV)
                            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        
                            model = simpleFF(X_CV.shape[1], hidden_size=hidden_size)
                            model.train()

                            optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)                                                                                               
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
                        y_pred = testSimpleFF(model, test_dataloader)
                        y_pred = np.squeeze(np.vstack(y_pred))                        
                        
                        test_MAE1 = np.mean(abs(y_pred - y_test))
                        test_MAE2 = None
                        test_r1 = stats.pearsonr(y_pred,y_test)[0]
                        test_r2 = None

                        test_age_1 = 100*y_test
                        test_age_2 = None

                        test_brainage_1 = y_pred # for single timepoint y is a vector
                        test_brainage_2 = y_pred

                else:
                    CV_scores, y_pred, test_MAE1, test_MAE2, test_r1, test_r2 = get_brain_age_perf(X_CV, y_CV, X_test, y_test, model_instance)
                    train_loss = np.mean(-1*CV_scores) #negative MSE
                    test_age_1 = y_test
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
    save_path = f"{args.save_path}/sim_perf_config_{config_idx}_iter_{it}.csv"

    config_df = pd.read_csv(config_file)

    traj_func = config_df.loc[config_idx,"traj_func"]
    roi_variation = config_df.loc[config_idx,"roi_variation"]
    subject_variation = config_df.loc[config_idx,"subject_variation"]
    followup_interval = config_df.loc[config_idx,"followup_interval"]
    data_aug = config_df.loc[config_idx,"data_aug"]

    perf_df = run(traj_func, roi_variation, subject_variation, n_samples_list, n_regions_list, data_aug, it)

    print(f"Saving simulation config: {config_idx} iter: {it} results at: {save_path}")
    perf_df.to_csv(save_path)