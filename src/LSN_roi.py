import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
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

    train_df = pd.read_csv(train_csv)
    train_eids = train_df["eid"]
    train_eids_avail = set(train_eids) & set(freesurfer_eids)
    train_df = train_df[train_df["eid"].isin(train_eids_avail)].reset_index()

    test_df = pd.read_csv(test_csv)
    test_eids = test_df["eid"]
    test_eids_avail = set(test_eids) & set(freesurfer_eids)
    test_df = test_df[test_df["eid"].isin(test_eids_avail)].reset_index()

    return train_df, test_df

class UKBB_ROI_Dataset(Dataset):
    ''' UKBB ROI Dataset comprsing FreeSurfer output
    '''

    def __init__(self, metadata_df, data_df, pheno_cols_ses2, pheno_cols_ses3, transform=None):
        self.metadata_df = metadata_df
        self.data_df = data_df 
        self.pheno_cols_ses2 = pheno_cols_ses2
        self.pheno_cols_ses3 = pheno_cols_ses3
        self.transform = transform
        
    def __len__(self):
        n_samples = len(self.metadata_df)
        return n_samples

    def __getitem__(self, idx):
        _df = self.metadata_df.copy()
        eid = _df.loc[idx,"eid"]
        age_ses2 = _df[_df["eid"]==eid]["age_at_ses2"].values[0]
        age_ses3 = _df[_df["eid"]==eid]["age_at_ses3"].values[0]    
        
        X_baseline = self.data_df[self.data_df["eid"]==eid][self.pheno_cols_ses2].values
        X_followup = self.data_df[self.data_df["eid"]==eid][self.pheno_cols_ses3].values
        
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

        return input_tensor, output_tensor


import torch

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches