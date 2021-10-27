import numpy as np
import pandas as pd
from datetime import datetime

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
                    

def get_brain_trajectory(n_timepoints, func_type, func_params):
    """ Generates ROI values over time based on a given polynomial model
    """
    t = np.arange(n_timepoints)
    if func_type == "exp":
        init_val = func_params["init_val"]
        decay = func_params["decay"]
        
        traj = init_val * np.exp(-1*(t/decay))

    elif func_type == "poly":
        init_val = func_params["init_val"]
        peak_val = func_params["peak_val"]
        time_shift = func_params["time_shift"]
        poly_order = func_params["poly_order"]

        traj = peak_val - ( (peak_val-init_val)*(t-time_shift)**poly_order ) / (time_shift**poly_order)

    else:
        print(f"Unknown function type: {func_type}")

    return traj

def get_traj_samples(traj, n_samples, intersubject_std, criterion="intercept"):
    """ Generates N trajectory samples by adding random factor
    """
    random_factor = intersubject_std*np.random.randn(n_samples)
    traj_sample = []
    for i in np.arange(n_samples):
        if criterion == "intercept":
            traj_sample.append(random_factor[i] + traj)
        else:
            print("to be implemented")

    return np.array(traj_sample)

def get_cross_sectional_samples(roi_list, followup_interval=0):
    """ Samples one timepoint per sample with replacement
    """
    n_samples, n_timepoints = roi_list[0].shape
    t = np.arange(n_timepoints-followup_interval)

    # sample only once and apply to all ROIs
    age_samples = np.random.choice(t,n_samples,replace=True)
    roi_sample_idx = list(zip(np.arange(n_samples), age_samples))

    roi_sampled_list = []
    for roi in roi_list:
        roi_sampled_list.append(roi[tuple(np.transpose(roi_sample_idx))])

    roi_sampled_array = np.vstack(roi_sampled_list).T

    # Followup visits --> same subjects and time-shifted ROI values
    followup_roi_sampled_array = None
    if followup_interval > 0:
        followup_roi_sampled_list = []
        followup_roi_sample_idx = list(zip(np.arange(n_samples), age_samples + followup_interval))
        for roi in roi_list:
            followup_roi_sampled_list.append(roi[tuple(np.transpose(followup_roi_sample_idx))])

        followup_roi_sampled_array = np.vstack(followup_roi_sampled_list).T

    return age_samples, roi_sampled_array, followup_roi_sampled_array

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

def augment_data(X_baseline, X_followup, y_baseline, y_followup, swap_only=True):
    """ Augments training (i.e. internal CV data) by swapping baseline and followup data
    """
    X_orig = np.hstack([X_baseline,X_followup])
    y_orig = np.vstack([y_baseline,y_followup]).T
    
    # swap timepoints
    X_swap = np.hstack([X_followup, X_baseline])
    y_swap = np.vstack([y_followup, y_baseline]).T
    
    # baseline only
    X_base = np.hstack([X_baseline, X_baseline])
    y_base = np.vstack([y_baseline, y_baseline]).T

    # followup only
    X_follow = np.hstack([X_followup, X_followup])
    y_follow = np.vstack([y_followup, y_followup]).T

    if swap_only:
        X_CV = np.vstack([X_orig,X_swap])
        y_CV = np.vstack([y_orig,y_swap])
    else:
        X_CV = np.vstack([X_orig,X_swap,X_base,X_follow])
        y_CV = np.vstack([y_orig,y_swap,y_base,y_follow])
    
    return X_CV, y_CV

    

## Torch 

# dataset
class SimpleSimDataset(Dataset):
    ''' Simulation dataset
    '''
    def __init__(self, X1, y1, transform=None):
        self.X1 = X1                
        self.y1 = y1
        
    def __len__(self):
        return len(self.y1)

    def __getitem__(self, idx):
        input1 = self.X1[idx]

        input1 = np.expand_dims(input1,0)

        output1 = self.y1[idx]/100
        output1 = np.expand_dims(output1,0)

        return (torch.tensor(input1,dtype=torch.float32), torch.tensor(output1,dtype=torch.float32))


class SimDataset(Dataset):
    ''' Simulation dataset
    '''
    def __init__(self, X1, X2, y1, y2, transform=None):
        self.X1 = X1        
        self.X2 = X2
        self.y1 = y1
        self.y2 = y2
        
    def __len__(self):
        return len(self.y1)

    def __getitem__(self, idx):
        input1 = self.X1[idx]
        input2 = self.X2[idx]

        input1 = np.expand_dims(input1,0)
        input2 = np.expand_dims(input2,0)

        output1 = self.y1[idx]/100
        output2 = self.y2[idx]/100
        output1 = np.expand_dims(output1,0)
        output2 = np.expand_dims(output2,0)

        return (torch.tensor(input1,dtype=torch.float32), torch.tensor(input2,dtype=torch.float32)), \
            (torch.tensor(output1,dtype=torch.float32), torch.tensor(output2,dtype=torch.float32))
