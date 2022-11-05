#!/bin/bash

# This is a script to run FSL fslreorient2std and FLIRT using Singularity container
# author: nikhil153
# date: 1 Aug 2022

if [ "$#" -ne 7 ]; then
    echo "Please specify BIDS_DIR, SUBJECT_ID, OUTPUT_DIR, REORIENT, DOF, and SINGULARITY_IMG"
    exit 1
fi

LOCAL_BIDS_DIR=$1
SUBJECT_ID=$2
SESSION=$3
LOCAL_OUTPUT_DIR=$4
REORIENT=$5
DOF=$6
SINGULARITY_IMG=$7

MNI152="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

SINGULARITY_BIDS_DIR="/bids_dir/"
SINGULARITY_OUTPUT_DIR="/fsl_dir/"

# inputs
preproc_T1="${SUBJECT_ID}_${SESSION}_desc-preproc_T1w.nii.gz"
preproc_T1_path="${LOCAL_BIDS_DIR}/${SUBJECT_ID}/${SESSION}/anat/${preproc_T1}"

mask_T1="${SUBJECT_ID}_${SESSION}_desc-brain_mask.nii.gz"
mask_T1_path="${LOCAL_BIDS_DIR}/${SUBJECT_ID}/${SESSION}/anat/${mask_T1}"

# intermediate outputs (mask + reorient)
preproc_masked_T1="${SUBJECT_ID}_${SESSION}_desc-preproc_masked.nii.gz"
preproc_masked_T1_path="${SINGULARITY_BIDS_DIR}/${SUBJECT_ID}/${SESSION}/anat/${preproc_masked_T1}"

brain_reoriented="${SUBJECT_ID}_${SESSION}_desc-reoriented.nii.gz"
brain_reoriented_path=${SINGULARITY_OUTPUT_DIR}/${brain_reoriented}

# registered output
# PMF: (fmriprep:Preproc+masked) + (FSL:Flirt)
PMF="${SUBJECT_ID}_${SESSION}_space-MNI152NLin6Sym_res-1_desc-PMF${DOF}_T1w.nii.gz"
PMF_path="${SINGULARITY_OUTPUT_DIR}/${PMF}"

# Run brain masking
echo "Applying brain mask"
python apply_brainmask.py --T1 $preproc_T1_path --mask $mask_T1_path

SINGULARITY_CMD="singularity run 
-B $LOCAL_BIDS_DIR:$SINGULARITY_BIDS_DIR \
-B $LOCAL_OUTPUT_DIR:$SINGULARITY_OUTPUT_DIR \
$SINGULARITY_IMG"

# Reorient
if [[ "$REORIENT" == "1" ]]; then
    fslreorient2std_CMD="fslreorient2std $preproc_masked_T1_path $brain_reoriented_path"

    echo "Running fslreorient2std before registration"
    echo ""

    EVAL_CMD="${SINGULARITY_CMD} ${fslreorient2std_CMD}"
    
    echo Commandline: $EVAL_CMD
    eval $EVAL_CMD
    
    in_path=$brain_reoriented_path

else
    in_path=$preproc_masked_T1_path
fi

# FLIRT
echo ""
echo "Running FLIRT with $DOF DOF" 
echo ""
flirt_CMD="flirt -in $in_path -ref $MNI152 -dof $DOF -out $PMF_path -omat T1toMNIlin.mat"

EVAL_CMD="${SINGULARITY_CMD} ${flirt_CMD}"
echo Commandline: $EVAL_CMD
eval $EVAL_CMD

exitcode=$?
