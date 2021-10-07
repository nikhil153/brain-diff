import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import numpy as np
import pandas as pd
from models import dp_utils as dpu

class twinSFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(twinSFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x1, x2):
        out1 = list()
        x1_f = self.feature_extractor(x1)
        x1 = self.classifier(x1_f)
        x1 = F.log_softmax(x1, dim=1)
        out1.append(x1)

        out2 = list()
        x2_f = self.feature_extractor(x2)
        x2 = self.classifier(x2_f)
        x2 = F.log_softmax(x2, dim=1)
        out2.append(x2)

        return out1, out2


def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

class UKBBDataset(Dataset):
    '''
        root_dir is the UKBB imaging directory
        img_subdir is the path to the T1 image from subject dir
        metadata_csv is a csv with age info
        n_samples is the size of the train set and the validation set combined
        transform is any image transformations
    '''
    def __init__(self, root_dirs, img_subdirs, metadata_csv, transform=None):
        self.root_dirs = root_dirs  
        self.img_subdirs = img_subdirs
        self.metadata_csv = metadata_csv
        self.transform = transform

    def __len__(self):
        ukbb_metadata = pd.read_csv(self.metadata_csv)
        return len(ukbb_metadata)

    def __getitem__(self, idx):
        inputs = None
        outputs = None
        crop_shape = (160, 192, 160)
        
        # Age 
        ukbb_metadata = pd.read_csv(self.metadata_csv)   
        eid = ukbb_metadata.loc[idx,"eid"]
        age_ses2 = ukbb_metadata[ukbb_metadata["eid"]==eid]["age_at_ses2"].values[0]
        age_ses3 = ukbb_metadata[ukbb_metadata["eid"]==eid]["age_at_ses3"].values[0]

        # Transforming the age to soft label (probability distribution)
        bin_range = [42,82]
        bin_step = 1
        sigma = 1
        age_ses2_soft, bc = dpu.num2vect(age_ses2, bin_range, bin_step, sigma)
        age_ses3_soft, bc = dpu.num2vect(age_ses3, bin_range, bin_step, sigma)
       
        # Sample subject needs to be in the MNI space
        subject_id = f"sub-{eid}"
        ses2_root_dir = self.root_dirs[0]
        ses3_root_dir = self.root_dirs[1]
        ses2_subdir = self.img_subdirs[0]
        ses3_subdir = self.img_subdirs[1]

        try: 
            # ses-2 image
            subject_dir = f"{ses2_root_dir}{subject_id}/{ses2_subdir}/"
            T1_mni = f"{subject_dir}T1_brain_to_MNI.nii.gz"
            img1 = nib.load(T1_mni).get_fdata()

            # ses-3 image # sub-1004084_ses-3
            subject_dir = f"{ses3_root_dir}{subject_id}/{ses3_subdir}/"
            T1_mni = f"{subject_dir}{subject_id}_ses-3_T1_brain_to_MNI.nii.gz"
            img2 = nib.load(T1_mni).get_fdata()

        # in case the path doesn't exist for MNI image
        # pick a eid that works
        except:
            print(f"{subject_id} has a missing scan. Using a fall-back sample : sub-1004084")
            eid = 1004084
            subject_id = f"sub-{eid}"

            # ses-2 image
            subject_dir = f"{ses2_root_dir}{subject_id}/{ses2_subdir}/"
            T1_mni = f"{subject_dir}T1_brain_to_MNI.nii.gz"
            img1 = nib.load(T1_mni).get_fdata()

            # ses-3 image
            subject_dir = f"{ses3_root_dir}{subject_id}/{ses3_subdir}/"
            T1_mni = f"{subject_dir}{subject_id}_ses-3_T1_brain_to_MNI.nii.gz"
            img2 = nib.load(T1_mni).get_fdata()

            # Age
            age_ses2 = ukbb_metadata[ukbb_metadata["eid"]==eid]["age_at_ses2"].values[0]
            age_ses3 = ukbb_metadata[ukbb_metadata["eid"]==eid]["age_at_ses3"].values[0]
            age_ses2_soft, bc = dpu.num2vect(age_ses2, bin_range, bin_step, sigma)
            age_ses3_soft, bc = dpu.num2vect(age_ses3, bin_range, bin_step, sigma)

        # minor preproc
        img1 = img1/img1.mean()
        img1 = crop_center(img1, crop_shape)
        img1 = np.expand_dims(img1,0)
    
        img2 = img2/img2.mean()        
        img2 = crop_center(img2, crop_shape)
        img2 = np.expand_dims(img2,0)

        print(f"eid: {eid}, age(s): {age_ses2,age_ses3}")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
    
        inputs = (torch.tensor(img1,dtype=torch.float32), torch.tensor(img2,dtype=torch.float32))
        outputs = (torch.tensor(age_ses2_soft, dtype=torch.float32), torch.tensor(age_ses3_soft, dtype=torch.float32))
        return inputs, outputs

# Toy network for testing siamese arch
class LSN(nn.Module):
    def __init__(self):
        super(LSN, self).__init__()
        
        # Conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv3d(1, 2, 5) 
        self.conv2 = nn.Conv3d(2, 2, 5)  
    
        self.bn1 = nn.BatchNorm3d(2)
        self.bn2 = nn.BatchNorm3d(2)
    
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
    
        # number of fc nodes = out_channels*((in_dim - kernel_size + 1 )/max_pool_dim)^3
        # crop_shape = (160, 192, 160)

        # after first conv + maxpool
        # 2*(((160 - 5 + 1)/4) * ((192 - 5 + 1)/4) * ((160 - 5 + 1)/4))
        # = 2 * (39 * 47 * 39)
        # after second conv + maxpool
        # 2 * ((39 - 5 + 1)//2) * ((47 - 5 + 1)//2) * ((39 - 5 + 1)//2)
        # = 2 * (17 * 21 * 17)
        
        self.fc_nodes = 2 * (17 * 21 * 17)
        self.fc1 = nn.Linear(self.fc_nodes, 128)
        self.fcOut = nn.Linear(128, 2)

        self.sigmoid = nn.Sigmoid()
    
    def convs(self, x):
        # out_dim = in_dim - kernel_size + 1  
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 4)
        
        x = F.relu(self.bn2(self.conv2(x)))        
        x = F.max_pool3d(x, 2)
        
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        print("conv shapes")
        print(f"x1 shape: {x1.shape}")
        x1 = x1.view(-1, self.fc_nodes)
        print("flattened shapes")
        print(f"x1 shape: {x1.shape}")
        x1 = self.sigmoid(self.fc1(x1))

        x2 = self.convs(x2)
        print("\nconv shapes")
        print(f"x2 shape: {x2.shape}")
        x2 = x2.view(-1, self.fc_nodes)
        print("flattened shapes")
        print(f"x2 shape: {x2.shape}")
        x2 = self.sigmoid(self.fc1(x2))

        print("\nfc shapes")
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        print(f"x1 max: {torch.max(x1)}, x2 max: {torch.max(x2)}")
        # x = torch.abs(x1 - x2)
        
        x = x1-x2
        x = self.fcOut(x)
        return x
