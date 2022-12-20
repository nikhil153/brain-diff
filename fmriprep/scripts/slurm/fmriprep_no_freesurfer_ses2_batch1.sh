#!/bin/bash

#SBATCH -J ukbb_ses-2_no_freesurfer_batch1
#SBATCH --time=12:00:00
#SBATCH --account=def-jbpoline
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
# Outputs ----------------------------------
#SBATCH -o ./slurm_logs/%x-%A-%a_%j.out
#SBATCH -e ./slurm_logs/%x-%A-%a_%j.err
#SBATCH --mail-user=nikhil.bhagwat@mcgill.ca
#SBATCH --mail-type=ALL
# ------------------------------------------

#SBATCH --array=801-1200

BIDS_DIR="/home/nikhil/scratch/ukbb_processing/bids/"
SUBJECT_LIST="/home/nikhil/scratch/ukbb_processing/subject_ids/ukbb_participant_ids_ses2-3.txt"
WD_DIR="/home/nikhil/scratch/ukbb_processing/derivatives/fmriprep/no_freesurfer/ses-2/"
TAR_DIR="/project/def-jbpoline/nikhil/ukbb_processing/derivatives/freesurfer-6.0.1/tars/no_freesurfer/"

echo "Starting task $SLURM_ARRAY_TASK_ID"
SUB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: ${SUB_ID}"

module load singularity/3.8
../fmriprep_anat_no_freesurfer_20.2.7.sh ${BIDS_DIR} ${WD_DIR} ${SUB_ID} ${TAR_DIR}
