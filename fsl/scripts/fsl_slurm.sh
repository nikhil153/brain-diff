#!/bin/bash

#SBATCH -J adni123_bl_fsl
#SBATCH --time=00:55:00
#SBATCH --account=def-jbpoline
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
# Outputs ----------------------------------
#SBATCH -o ./slurm_logs/%x-%A-%a_%j.out
#SBATCH -e ./slurm_logs/%x-%A-%a_%j.err
#SBATCH --mail-user=nikhil.bhagwat@mcgill.ca
#SBATCH --mail-type=ALL
# ------------------------------------------

#SBATCH --array=601-1106

BIDS_DIR="/home/nikhil/scratch/adni_processing/adni123_bl_m24_July_2022/derivatives/bl/fmriprep"
SUBJECT_LIST="/home/nikhil/scratch/my_repos/brain-diff/metadata/adni/subject_lists/adni123_fsl_subject_ids.txt"
OUT_DIR="/home/nikhil/scratch/adni_processing/adni123_bl_m24_July_2022/derivatives/bl/fsl"

echo "Starting task $SLURM_ARRAY_TASK_ID"
SUB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: ${SUB_ID}"

REORIENT="1"
DOF="6"

module load singularity/3.8
source /home/nikhil/projects/def-jbpoline/nikhil/env/green_compute/bin/activate

./run_FSL.sh ${BIDS_DIR} ${SUB_ID} ${OUT_DIR} $REORIENT $DOF /home/nikhil/scratch/my_containers/Singularity.fsl.sif
