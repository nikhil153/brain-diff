
import numpy as np
import nibabel as nib
import os

import argparse


HELPTEXT = """
Script to mask T1w images (expects BIDS filename format)
e.g. sub-ADNI941S6580_ses-bl_desc-preproc_T1w.nii.gz, sub-ADNI941S6580_ses-bl_desc-brain_mask.nii.gz
"""

parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--T1', dest='T1', 
                    help='Path to T1w image')
parser.add_argument('--mask', dest='mask', 
                    help='Path to mask image')

args = parser.parse_args()

T1_basename = os.path.basename(args.T1)
T1_dir = args.T1.rsplit("/",1)[0]
sub_id, ses_id, _ = T1_basename.split("_",2)
masked_T1_path = f"{T1_dir}/{sub_id}_{ses_id}_desc-preproc_masked.nii.gz"

T1 = nib.load(args.T1)
T1_affine = T1.affine
T1_data = T1.get_fdata()
brainmask_data = nib.load(args.mask).get_fdata()
masked_T1_data = brainmask_data * T1_data

masked_T1 = nib.Nifti1Image(masked_T1_data, T1.affine, T1.header)

nib.save(masked_T1, masked_T1_path)