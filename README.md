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


### UKB data wrangling

#### Step 1
- copy files from squashfs on Beluga
Ses-2 (n=40681): <neurohub_ukbb_t1w_bids_derivatives.squashfs>:/neurohub/ukbb/imaging/T1
Ses-3 (n=3208): <neurohub_ukbb_t1w_ses3_0_derivatives.squashfs>:/neurohub/ukbb/imaging

#### Step 2
```
## move them in psudo-bids
for i in `ls | grep sub- | grep -v json`; do 
    mkdir -p ../`echo $i | cut -d "_" -f1`/ses-2/anat; 
    mv `echo $i | cut -d "_" -f1`* ../`echo $i | cut -d "_" -f1`/ses-2/anat/;  
done
```

### ADNI data wrangling
- use src/generate_adni_bids.py

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
