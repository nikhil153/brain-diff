import numpy as np
import pandas as pd
from datetime import datetime

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
                    

def get_brain_trajectory(n_timepoints, init_val, peak_val, time_shift, poly_order):
    """ Generates ROI values over time based on a given polynomial model
    """
    t = np.arange(n_timepoints)
    traj = peak_val - ( init_val*(t-time_shift)**poly_order ) / (time_shift**poly_order)
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
            followup_roi_sampled_list.append(roi[tuple(np.transpose(roi_sample_idx))])

        followup_roi_sampled_array = np.vstack(followup_roi_sampled_list).T

    return age_samples, roi_sampled_array, followup_roi_sampled_array

def get_brain_age_perf(X_CV, y_CV, X_test, y_test, model, cv=2):
    """ Compute CV score and heldout sample MAE and correlation
    """
    pipeline = Pipeline([("brainage_model", model)])
    pipeline.fit(X_CV, y_CV)

    # Evaluate the models using crossvalidation
    CV_scores = cross_val_score(pipeline, X_CV, y_CV,
                                scoring="neg_mean_squared_error", cv=cv)

    ## predict on held out test
    y_pred = pipeline.predict(X_test)
    if y_test.ndim == 1:
        test_MAE = 100*(abs(y_pred - y_test)) #Scale age
        test_r = stats.pearsonr(y_pred,y_test)[0]
    else:
        test_MAE1 = 100*abs(y_pred[:,0] - y_test[:,0]) #Scale age
        test_r1 = stats.pearsonr(y_pred[:,0],y_test[:,0])[0]
        test_MAE2 = 100*abs(y_pred[:,1] - y_test[:,1]) #Scale age
        test_r2 = stats.pearsonr(y_pred[:,1],y_test[:,1])[0]
        
    return CV_scores, 100*y_pred, test_MAE1, test_MAE2, test_r1, test_r2


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

        output1 = self.y1[idx]
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

        output1 = self.y1[idx]
        output2 = self.y2[idx]
        output1 = np.expand_dims(output1,0)
        output2 = np.expand_dims(output2,0)

        return (torch.tensor(input1,dtype=torch.float32), torch.tensor(input2,dtype=torch.float32)),(torch.tensor(output1,dtype=torch.float32), torch.tensor(output2,dtype=torch.float32))


class simpleFF(nn.Module):
    def __init__(self, input_size,hidden_size,output_size=1):
        super(simpleFF, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
    
        self.fcOut = nn.Linear(self.hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.fc3(self.fc2(self.fc1(x)))
        x = self.sigmoid(self.fcOut(x))
        return x

# Toy network for testing siamese arch
class LSN(nn.Module):
    def __init__(self, input_size,hidden_size,output_size=1):
        super(LSN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(2*self.hidden_size, self.hidden_size) #concat
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, self.hidden_size)

        self.drop = nn.Dropout(p=0.2)
    
        self.fcOut = nn.Linear(self.hidden_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        # lower twin branches
        x1 = self.drop(self.fc2(self.fc1(x1)))
        x2 = self.drop(self.fc2(self.fc1(x2)))

        # middle concat
        x = torch.cat([x1,x2],dim=2)
        x = self.fc3(x)

        # upper splits
        x3 = self.fc4(x) 
        x4 = self.fc5(x) 
        
        # predict (don't want sigmoid!)
        x3 = self.relu(self.fcOut(x3))
        x4 = self.relu(self.fcOut(x4))

        x_out = torch.cat([x3,x4],dim=2)

        return x_out

def twinLoss(x1,x2,y1,y2,loss_func):
    """Returns twin loss for a given loss func
    """
    loss1 = loss_func(x1,y1)
    loss2 = loss_func(x2,y2)
    loss = 0.5*(loss1 + loss2)
    return loss


def trainSimpleFF(model, train_dataloader, optimizer, criterion, n_epochs):
    batch_loss_list = []
    epoch_loss_list = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        # print("Starting epoch " + str(epoch+1))
        # print(f"len dataloader: {len(train_dataloader)}")
        for inputs, outputs in train_dataloader:
            img1 = inputs[0]
            age_at_ses2 = outputs[0]

            img1 = img1.to(device)            
            age_at_ses2 = age_at_ses2.to(device)
                                
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            preds = model(img1)            
            loss = criterion(preds, age_at_ses2)             
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_loss_list.append(loss.item())
        
        
        epoch_loss = running_loss/len(train_dataloader)
        print(f"epoch {epoch} loss: {epoch_loss:5.4f}")
        epoch_loss_list.append(epoch_loss)

    ## loss df
    batch_loss_df = pd.DataFrame()
    batch_loss_df["batch_loss"] = batch_loss_list

    epoch_loss_df = pd.DataFrame()
    epoch_loss_df["epoch_loss"] = epoch_loss_list

    return model, batch_loss_df, epoch_loss_df


def train(model, train_dataloader, optimizer, criterion, n_epochs):
    batch_loss_list = []
    epoch_loss_list = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        # print("Starting epoch " + str(epoch+1))
        # print(f"len dataloader: {len(train_dataloader)}")
        for inputs, outputs in train_dataloader:
            img1 = inputs[0]
            img2 = inputs[1]
    
            age_at_ses2 = outputs[0]
            age_at_ses3 = outputs[1]

            img1 = img1.to(device)
            img2 = img2.to(device)
            age_at_ses2 = age_at_ses2.to(device)
            age_at_ses3 = age_at_ses3.to(device)
                        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            preds = model(img1, img2)
            
            loss = twinLoss(preds[:,:,0], preds[:,:,1], age_at_ses2, age_at_ses3, criterion) 
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_loss_list.append(loss.item())
        
        
        epoch_loss = running_loss/len(train_dataloader)
        # print(f"epoch {epoch} loss: {epoch_loss:5.4f}")
        epoch_loss_list.append(epoch_loss)
    print(f"epoch {epoch} loss: {epoch_loss:5.4f}")

    ## loss df
    batch_loss_df = pd.DataFrame()
    batch_loss_df["batch_loss"] = batch_loss_list

    epoch_loss_df = pd.DataFrame()
    epoch_loss_df["epoch_loss"] = epoch_loss_list

    return model, batch_loss_df, epoch_loss_df


def testSimpleFF(model, test_dataloader):
    with torch.no_grad():
        batch_loss_list = []
        for inputs, outputs in test_dataloader:
            img1 = inputs[0]            
            age_at_ses2 = outputs[0]
            img1 = img1.to(device)        
            age_at_ses2 = age_at_ses2.to(device) 

            preds = model(img1) 
                        
            criterion = nn.L1Loss()
            loss = criterion(preds, age_at_ses2)

            batch_loss_list.append(loss.item())
        
    return batch_loss_list

def test(model, test_dataloader, criterion=nn.L1Loss()):
    with torch.no_grad():
        loss1_list = []
        loss2_list = []
        batch_pred_list = []
        for inputs, outputs in test_dataloader:
            img1 = inputs[0]
            img2 = inputs[1]

            age_at_ses2 = 100*outputs[0] #Scale age
            age_at_ses3 = 100*outputs[1] #Scale age

            img1 = img1.to(device)
            img2 = img2.to(device)
            age_at_ses2 = age_at_ses2.to(device) 
            age_at_ses3 = age_at_ses3.to(device) 
            
            preds = 100*model(img1, img2) #Scale age
            batch_pred_list.append(preds.detach().numpy())

            loss1 = criterion(preds[:,:,0],age_at_ses2)
            loss2 = criterion(preds[:,:,1],age_at_ses3)
            
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())

        
    return batch_pred_list, loss1_list, loss2_list