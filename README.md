# brain-diff

## Goals: 
1) Brainage prediction with two timepoints
2) Brainage Biomarker for AD and PD 

### Related work
    - Peng H, Gong W, Beckmann CF, Vedaldi A, Smith SM. Accurate brain age prediction with lightweight deep neural networks. Med Image Anal. 2021. [Paper](https://doi.org/10.1016/j.media.2020.101871), [Code](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain)
    - Jonsson, B.A., Bjornsdottir, G., Thorgeirsson, T.E. et al. Brain age prediction using deep learning uncovers associated sequence variants. Nat Commun 10, 5409 (2019). [Paper](https://www.nature.com/articles/s41467-019-13163-9), Code on request.
    - Baecker L, Garcia-Dias R, Vieira S, Scarpazza C, Mechelli A. Machine learning for brain age prediction: Introduction to methods and clinical applications. EBioMedicine. 2021 [Paper](https://pubmed.ncbi.nlm.nih.gov/34614461/)
    - Leonardsen EH, Peng H, ... Wang Y. Deep neural networks learn general and clinically relevant representations of the ageing brain. Neuroimage. 2022. [Paper](https://pubmed.ncbi.nlm.nih.gov/35462035/)


### Datasets
    - UKBB
    - ADNI
    - PPMI

### UKB data wrangling
    - Copy files from squashfs on Beluga
    - Organize them in psudo-bids

```
for i in `ls | grep sub- | grep -v json`; do 
    mkdir -p ../`echo $i | cut -d "_" -f1`/ses-2/anat; 
    mv `echo $i | cut -d "_" -f1`* ../`echo $i | cut -d "_" -f1`/ses-2/anat/;  
done
```

### ADNI data wrangling
    - use src/generate_adni_bids.py

### MR preprocessing
    - fmriprep anat workflow
        - template spaces: `MNI152NLin2009cSym_res-1`, `MNI152NLin6Sym_res-1`, `MNI152Lin_res-1`
    - freesufer 6.0.1
        - DKT (n_rois: 31x2) cortical thickness and ASEG volumes

### Experiments: model training with controls (UKBB)
    - input_visit --> output_visit
        1. Baseline --> Baseline
        2. Baseline + Followup --> Baseline
        3. Baseline + Followup --> Baseline + Followup
    - features + models
        1. DKT (Ridge, RF)
        2. T1w normalized to the MNI template(s) (SFCN, LSN)
    - nulls
        1. Median (+2) age prediction
    - perf metrics
        1. mean abs error
        2. pearson's r
        3. temporal consistency 

### Experiments: model biases on control cohorts
    - age vs brainage_error bias (effectiveness of linear correction)
    - short vs long visit_delta: UKB longterm cohort (FU - BL > 3yr)
    - study+scanner variation: ADNI, PPMI control cohorts

### Experiments: brainage gap 
    - Single number: Disease stages vs study-specific controls vs long_visit UKBB vs short_visit ukbb
        - Note even with two visits only BL brainage value is likely to be useful. 
    - DeepNet representations: Richer constellation / clusterring of subjects from model embeddings 

--- 

### Simulations:
    - Simple interactive runs: notebooks/7_brain_diff_sim.ipynb
    - Batch runs: src/run_simul.py

### SFCN depoyment on new data:
    - src/run_SFCN.py
    