# brain-diff

## Goal: Brainage prediction with two timepoints

### Replication
    - [Paper](https://doi.org/10.1016/j.media.2020.101871): Accurate brain age prediction with lightweight deep neural networks Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith Medical Image Analysis (2021)
    - Code [repo](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain)

### Datasets
    - UKBB: notebooks/1_ukb_follow_up.ipynb
    - ADNI: notebooks/2_adni_follow_up.ipynb
    - Simulations: notebooks/7_brain_diff_sim.ipynb

### Results
    - Brainage replication: notebooks/4_brain_age.ipynb
    - Simulation: notebooks/8_brain_diff_sim_results.ipynb



## Run instructions
### Simulations:
    - Simple interactive runs: notebooks/7_brain_diff_sim.ipynb
    - Batch runs: src/run_simul.py

### SFCN replication:
    - src/run_SFCN.py
    
### slurm setup for training LSN
1. module load singularity/3.8
2. singularity shell --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1w_bids_derivatives.squashfs:ro --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1w_bids_derivatives_ses3_0_bids.squashfs /home/nikhil/scratch/FastSurfer.sif
3. ./run_LSN.sh
