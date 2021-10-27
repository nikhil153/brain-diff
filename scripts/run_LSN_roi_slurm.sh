#!/bin/bash
#SBATCH --account=rrg-jbpoline
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=32G               # memory (per node)
#SBATCH --time=0-23:00            # time (DD-HH:MM)
#SBATCH --job-name=brain_diff_LSN_roi
#SBATCH --output=logs/%x-%j.out
#SBATCH --array=0-2

echo "Starting task $SLURM_ARRAY_TASK_ID"

CONFIG_ID=$SLURM_ARRAY_TASK_ID
RUN_ID=$1
DATA_DIR="/data"
RESULTS_DIR="/results"

module load singularity/3.8
singularity exec -B /home/nikhil/scratch/brain_diff/LSN_roi/results:/$RESULTS_DIR \
		 -B /home/nikhil/scratch/brain_diff/LSN_roi/data:/$DATA_DIR \
            /home/nikhil/scratch/FastSurfer.sif ./run_LSN_roi.sh $RUN_ID $CONFIG_ID $DATA_DIR $RESULTS_DIR
