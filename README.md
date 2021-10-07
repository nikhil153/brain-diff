# brain-diff

### slurm setup for training LSN
1. module load singularity/3.8
2. singularity shell --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1w_bids_derivatives.squashfs:ro --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1w_bids_derivatives_ses3_0_bids.squashfs /home/nikhil/scratch/FastSurfer.sif
3. ./run_LSN.sh
