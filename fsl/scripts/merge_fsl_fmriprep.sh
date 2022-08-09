#!/bin/bash

# This is a script to to merge FSL output (i.e. FLIRT) into fmriprep dir
# author: nikhil153
# date: 1 Aug 2022

if [ "$#" -ne 4 ]; then
    echo "Please specify FSL_DIR, FMRIPREP_DIR, SES_ID, AND SUBJECT_ID"
    exit 1
fi

FSL_DIR=$1
FMRIPREP_DIR=$2
SES_ID=$3
SUBJECT_ID=$4

rsync -av ${FSL_DIR}/${SUBJECT_ID}_${SES_ID}* ${FMRIPREP_DIR}/${SUBJECT_ID}/${SES_ID}/anat/