import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random


def get_ML_dataframes(usecols, freesurfer_csv, train_csv, test_csv):
    ''' Create train and test datasets based on available freesurfer data
    '''
    freesurfer_df = pd.read_csv(freesurfer_csv, usecols=usecols)

    # Remove eids with missing 2nd or 3rd ses data
    eid_missing_data = freesurfer_df[freesurfer_df.isna().any(axis=1)]["eid"].values
    print(f"number participants missing 2nd or 3rd ses freesurfer data: {len(eid_missing_data)}")
    freesurfer_eids = freesurfer_df[~freesurfer_df["eid"].isin(eid_missing_data)]["eid"].values

    freesurfer_df = freesurfer_df[~freesurfer_df["eid"].isin(eid_missing_data)]

    train_df = pd.read_csv(train_csv)
    train_eids = train_df["eid"]
    train_eids_avail = set(train_eids) & set(freesurfer_eids)
    train_df = pd.merge(train_df, freesurfer_df, on="eid", how="inner")

    test_df = pd.read_csv(test_csv)
    test_eids = test_df["eid"]
    test_eids_avail = set(test_eids) & set(freesurfer_eids)
    test_df = pd.merge(test_df, freesurfer_df, on="eid", how="inner")

    return train_df, test_df

class UKBB_ROI_Dataset(Dataset):
    ''' UKBB ROI Dataset comprsing FreeSurfer output
    '''

    def __init__(self, data_df, pheno_cols_ses2, pheno_cols_ses3, transform=None):
        self.data_df = data_df 
        self.pheno_cols_ses2 = pheno_cols_ses2
        self.pheno_cols_ses3 = pheno_cols_ses3
        self.transform = transform
        
    def __len__(self):
        n_samples = len(self.data_df)
        return n_samples

    def __getitem__(self, idx):
        _df = self.data_df.copy()
        eid = _df.loc[idx,"eid"]
        age_ses2 = _df[_df["eid"]==eid]["age_at_ses2"].values[0]
        age_ses3 = _df[_df["eid"]==eid]["age_at_ses3"].values[0]    
        
        X_baseline = _df[_df["eid"]==eid][self.pheno_cols_ses2].values
        X_followup = _df[_df["eid"]==eid][self.pheno_cols_ses3].values

        # input1 = np.expand_dims(input1,0)
        # input2 = np.expand_dims(input2,0)

        y_baseline = age_ses2/100
        y_followup = age_ses3/100
        y_baseline = np.expand_dims(y_baseline,0)
        y_followup = np.expand_dims(y_followup,0)

        if self.transform == "random_swap": #used during training
            p = random.uniform(0, 1)
            if p > 0.5:
                input_tensor = (torch.tensor(X_followup,dtype=torch.float32), torch.tensor(X_baseline,dtype=torch.float32))
                output_tensor = (torch.tensor(y_followup,dtype=torch.float32), torch.tensor(y_baseline,dtype=torch.float32))
            else:
                input_tensor = (torch.tensor(X_baseline,dtype=torch.float32), torch.tensor(X_followup,dtype=torch.float32))
                output_tensor = (torch.tensor(y_baseline,dtype=torch.float32), torch.tensor(y_followup,dtype=torch.float32))

        elif self.transform == "swap": #used during testing
            input_tensor = (torch.tensor(X_followup,dtype=torch.float32), torch.tensor(X_baseline,dtype=torch.float32))
            output_tensor = (torch.tensor(y_followup,dtype=torch.float32), torch.tensor(y_baseline,dtype=torch.float32))

        else:
            input_tensor = (torch.tensor(X_baseline,dtype=torch.float32), torch.tensor(X_followup,dtype=torch.float32))
            output_tensor = (torch.tensor(y_baseline,dtype=torch.float32), torch.tensor(y_followup,dtype=torch.float32))

        return eid, input_tensor, output_tensor
