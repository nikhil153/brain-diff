#!/bin/bash
#SBATCH --account=rrg-jbpoline
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G               # memory (per node)
#SBATCH --time=0-47:00            # time (DD-HH:MM)
#SBATCH --job-name=LSN_run_0
#SBATCH --output=logs/%x-%j.out

module load singularity/3.8

echo "training using CPU"

singularity exec --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1w_bids_derivatives.squashfs:ro \
		 --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1w_bids_derivatives_ses3_0_bids.squashfs:ro \
		 /home/nikhil/scratch/FastSurfer.sif \
		./run_LSN.sh
