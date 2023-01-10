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

# Torch Models
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
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.fc3(self.fc2(self.fc1(x))))
        x = self.relu(self.fcOut(x))
        return x

# Toy network for testing siamese arch
class LSN_FF_Linear(nn.Module):
    def __init__(self, input_size,hidden_size,output_size=1):
        super(LSN_FF_Linear, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(2*self.hidden_size, self.hidden_size) #concat
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6 = nn.Linear(self.hidden_size, self.hidden_size)

        self.drop = nn.Dropout(p=0.2)
    
        self.fcOut1 = nn.Linear(self.hidden_size, output_size)
        self.fcOut2 = nn.Linear(self.hidden_size, output_size)

        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        # print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")

        # lower twin branches
        x1 = self.fc2(self.fc1(x1))
        x2 = self.fc2(self.fc1(x2))
        
        # Joint features the branches from here on
        x = torch.cat([x1,x2],dim=2)
        x = self.fc3(x)
        x = self.fc4(x)
        
        # Upper separate branches
        x3 = self.fc5(x)
        x4 = self.fc6(x)

        # predict (don't want sigmoid or Relu!)
        x3 = self.fcOut1(x3)
        x4 = self.fcOut2(x4)
    
        x_out = torch.cat([x3,x4],dim=2)

        return x_out

class LSN_FF(nn.Module):
    def __init__(self, input_size,hidden_size,output_size=1):
        super(LSN_FF, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(2*self.hidden_size, self.hidden_size) #concat
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6 = nn.Linear(self.hidden_size, self.hidden_size)

        self.drop = nn.Dropout(p=0.2)
    
        self.fcOut1 = nn.Linear(self.hidden_size, output_size)
        self.fcOut2 = nn.Linear(self.hidden_size, output_size)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        # lower common (i.e. twin) branches
        x1 = self.relu(self.fc1(x1))
        x2 = self.relu(self.fc1(x2))

        # x1 = self.relu(self.fc2(x1))
        # x2 = self.relu(self.fc2(x2))

        # middle concat
        # print(f"x1,x2 shapes: {x1.shape}, {x2.shape}")
        x = torch.cat([x1,x2],dim=2)
        
        # print(f"concat shape: {x.shape}")
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        
        # upper splits
        x3 = self.relu(self.fc5(x))
        x4 = self.relu(self.fc6(x))
        
        # predict (don't want sigmoid!)
        x3 = self.fcOut1(x3)
        x4 = self.fcOut2(x4)
        # x3 = self.relu(self.fcOut(x3))
        # x4 = self.relu(self.fcOut(x4))

        x_out = torch.cat([x3,x4],dim=2)
        # x_out = self.relu(self.fcOut(x))

        return x_out

def twinLoss(x1,x2,y1,y2,loss_func):
    """Returns twin loss for a given loss func
    """
    loss1 = loss_func(x1,y1)
    loss2 = loss_func(x2,y2)
    loss = 0.5*(loss1 + loss2)
    return loss

# Single visit 
def trainSimpleFF(model, train_dataloader, optimizer, criterion, n_epochs):
    batch_loss_list = []
    epoch_loss_list = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        # print("Starting epoch " + str(epoch+1))
        # print(f"len dataloader: {len(train_dataloader)}")
        for inputs, outputs in train_dataloader:
            img1 = inputs[:,0]
            age_at_ses2 = outputs

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
        # print(f"epoch {epoch} loss: {epoch_loss:5.4f}")
        epoch_loss_list.append(epoch_loss)

    ## loss df
    batch_loss_df = pd.DataFrame()
    batch_loss_df["batch_loss"] = batch_loss_list

    epoch_loss_df = pd.DataFrame()
    epoch_loss_df["epoch_loss"] = epoch_loss_list

    return model, batch_loss_df, epoch_loss_df

def testSimpleFF(model, test_dataloader, criterion=nn.L1Loss()):
    with torch.no_grad():
        batch_loss_list = []
        batch_pred_list = []
        for inputs, outputs in test_dataloader:
            img1 = inputs[:,0]            
            
            img1 = img1.to(device)        
            age_at_ses2 = 100*outputs.to(device) 

            preds = 100*model(img1) 
            batch_pred_list.append(preds.detach().numpy())
                                    
            loss = criterion(preds, age_at_ses2)
            batch_loss_list.append(loss.item())
        
    return batch_pred_list, batch_loss_list

# LSN
def train(model, train_dataloader, optimizer, criterion, n_epochs):
    batch_loss_list = []
    epoch_loss_list = []
    preds_df = pd.DataFrame()
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch+1))
        # print(f"len dataloader: {len(train_dataloader)}")
        batch_idx = 0
        for eids, inputs, outputs in train_dataloader:
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
            p_df = pd.DataFrame()
            p_df["eid"] = eids
            p_df["pred_baseline"] = np.vstack(preds.detach().numpy())[:,0]
            p_df["pred_followup"] = np.vstack(preds.detach().numpy())[:,1]
            p_df["epoch"] = epoch
            p_df["batch_idx"] = batch_idx
            
            preds_df = preds_df.append(p_df)
            
            # loss = criterion(preds[:,:,0],age_at_ses2)
            loss = twinLoss(preds[:,:,0], preds[:,:,1], age_at_ses2, age_at_ses3, criterion) 
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_loss_list.append(loss.item())
            # print(f"running loss: {running_loss}")
            batch_idx = batch_idx + 1
        
        
        epoch_loss = running_loss/len(train_dataloader)
        epoch_loss_list.append(epoch_loss)
        print(f"epoch {epoch} loss: {epoch_loss:5.4f}")

    print(f"epoch {epoch} loss: {epoch_loss:5.4f}")

    ## loss df
    batch_loss_df = pd.DataFrame()
    batch_loss_df["batch_loss"] = batch_loss_list

    epoch_loss_df = pd.DataFrame()
    epoch_loss_df["epoch_loss"] = epoch_loss_list

    return model, batch_loss_df, epoch_loss_df, preds_df


def test(model, test_dataloader, criterion=nn.L1Loss()):
    model.eval()
    with torch.no_grad():
        eid_list = []
        loss1_list = []
        loss2_list = []
        y_pred_list = []
        y_test_list = []
        for eid, inputs, outputs in test_dataloader:
            eid_list.append(eid.detach().numpy())
            img1 = inputs[0]
            img2 = inputs[1]

            age_at_ses2 = 100*outputs[0] 
            age_at_ses3 = 100*outputs[1] 

            y_test = (age_at_ses2,age_at_ses3)
            y_test_list.append(y_test)

            img1 = img1.to(device)
            img2 = img2.to(device)
            age_at_ses2 = age_at_ses2.to(device) 
            age_at_ses3 = age_at_ses3.to(device) 
            
            preds = 100*model(img1, img2) #Scale age
            y_pred_list.append(preds.detach().numpy())

            loss1 = criterion(preds[:,:,0],age_at_ses2)
            loss2 = criterion(preds[:,:,1],age_at_ses3)
            
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())

    return eid_list, y_test_list, y_pred_list, loss1_list, loss2_list