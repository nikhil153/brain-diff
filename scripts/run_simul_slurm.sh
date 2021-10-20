#!/bin/bash
#SBATCH --account=rrg-jbpoline
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=32G               # memory (per node)
#SBATCH --time=0-24:00            # time (DD-HH:MM)
#SBATCH --job-name=brain_diff_simul
#SBATCH --output=logs/%x-%j.out
#SBATCH --array=0-59

echo "Starting task $SLURM_ARRAY_TASK_ID"

CONFIG_ID=$SLURM_ARRAY_TASK_ID
RUN_ID=$1

module load singularity/3.8
singularity exec -B /home/nikhil/scratch/brain_diff/simulation/results:/results_dir /home/nikhil/scratch/FastSurfer.sif ./run_simul.sh $RUN_ID $CONFIG_ID
