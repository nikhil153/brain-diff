#!/bin/bash
#SBATCH --account=rrg-jbpoline
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=16G               # memory (per node)
#SBATCH --time=0-4:00            # time (DD-HH:MM)
#SBATCH --job-name=brain_age_prediction
#SBATCH --output=logs/%x-%j.out

if [ $# -eq 0 ] ; then
   echo "No arguments supplied"
   exit 1
fi


module load singularity/3.8

SES=$1
echo "predicting for $SES"

if [[ "$SES" == "ses-2" ]] ; then
   SQUASH="/project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1w_bids_derivatives.squashfs:ro"
   RUN_SCRIPT="run_SFCN_ses-2.sh"
else
   SQUASH="/project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1w_bids_derivatives_ses3_0_bids.squashfs:ro"
   RUN_SCRIPT="run_SFCN_ses-3.sh"
fi

singularity exec --overlay $SQUASH \
		 /home/nikhil/scratch/FastSurfer.sif \
		 $RUN_SCRIPT
