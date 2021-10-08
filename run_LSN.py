import os 
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from LSN import *
from models import dp_loss as dpl
from models import dp_utils as dpu

HELPTEXT = """
Script to train brain age LSN (i.e. twinSFCN) model 
Author: nikhil153
Date: Oct-5-2021
"""

parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--data_dirs', dest='data_dirs',  nargs='+',
                    default="/home/nikhil/projects/brain_changes/data/ukbb/imaging/ukbb_test_subject/",
                    help='two data dirs containing all the subjects for each timepoint')
                    
parser.add_argument('--img_subdirs', dest='img_subdirs', nargs='+',
                    default="ses-2/non-bids/T1/",
                    help='path from subject dir to T1 scan')

parser.add_argument('--metadata_csv', dest='metadata_csv', 
                    default="./metadata.csv",
                    help='metadata csv with eid and age columns')

parser.add_argument('--sfcn_ckpt', dest='sfcn_ckpt', 
                    default="models/run_20190719_00_epoch_best_mae.p",
                    help='pre-trained SFCN model weights')             

parser.add_argument('--batch_size', dest='batch_size', 
                    default=2,
                    help='batch size for training')

parser.add_argument('--n_epochs', dest='n_epochs', 
                    default=2,
                    help='n_epochs for training')

parser.add_argument('--save_path', dest='save_path', 
                    default="./",
                    help='path to save model checkpoint and train loss')

args = parser.parse_args()

def train(model, train_dataloader, optimizer, criterion, n_epochs):
    batch_loss_list = []
    epoch_loss_list = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch+1))
        print(f"len dataloader: {len(train_dataloader)}")
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
            
            loss = criterion(preds[0][0],preds[1][0],age_at_ses2,age_at_ses3) #criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # print statistics
            print(f"loss: {loss.item()}")
            running_loss += loss.item()
            batch_loss_list.append(loss.item())
        
        
        epoch_loss = running_loss/len(train_dataloader)
        print(f"epoch loss: {epoch_loss:3.2f}")
        epoch_loss_list.append(epoch_loss)

    ## loss df
    batch_loss_df = pd.DataFrame()
    batch_loss_df["batch_loss"] = batch_loss_list

    epoch_loss_df = pd.DataFrame()
    epoch_loss_df["epoch_loss"] = epoch_loss_list

    return model, batch_loss_df, epoch_loss_df

if __name__ == "__main__":
                   
    data_dirs = args.data_dirs
    sfcn_ckpt = args.sfcn_ckpt
    img_subdirs = args.img_subdirs
    metadata_csv = args.metadata_csv
    save_path = args.save_path
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)

    criterion = dpl.twin_KLDivLoss

    ## Dataloader
    batch_size = 1
    ukbb_dataset = UKBBDataset(data_dirs, img_subdirs, metadata_csv)
    train_dataloader = DataLoader(ukbb_dataset, batch_size=batch_size, shuffle=True)

    ## Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = twinSFCN() # alternative toy model: LSN()

    model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    # Original SFCN model
    if sfcn_ckpt.split("/",1)[1] == "run_20190719_00_epoch_best_mae.p":
        print("Using original SFCN checkpoint")
        model.load_state_dict(torch.load(sfcn_ckpt, map_location=torch.device('cpu')))
    else:
        print("Using fine-tuned LSN checkpoint")
        checkpoint = torch.load(sfcn_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    start_time = datetime.now()
    print(f"Start training at: {start_time}")
    model, batch_loss_df, epoch_loss_df = train(model,train_dataloader,optimizer,criterion,n_epochs)

    end_time = datetime.now()
    print(f"End training at: {end_time}")

    print(f"Saving model chech point and loss logs at: {save_path}")
    ## Save checkpoint
    ckpt_save_path = save_path + "lsn.ckpt"
    torch.save({'epoch': n_epochs,
                'model_state_dict': model.state_dict(),            
                }, ckpt_save_path)

    ## save_losses
    batch_loss_df.to_csv(save_path + "batch_loss.csv")
    epoch_loss_df.to_csv(save_path + "epoch_loss.csv")
