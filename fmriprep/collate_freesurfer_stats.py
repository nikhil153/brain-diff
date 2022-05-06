import numpy as np
import pandas as pd
import os
import glob
import argparse
from freesurfer_stats import CorticalParcellationStats

HELPTEXT = """
Script to parse and collate FreeSurfer stats files across subjects
Author: nikhil153
Date: May-5-2022
"""

# Sample cmd:
#  python collate_freesurfer_stats.py --stat_file aparc.DKTatlas.stats \
#                                     --stat_measure average_thickness_mm \
#                                     --fs_output_dir /home/nikhil/projects/QPN_processing/test_data/fmriprep/output/freesurfer-6.0.1/ \
#                                     --UKBB_dkt_ct_fields /home/nikhil/projects/brain_changes/brain-diff/metadata/UKBB_DKT_CT_Fields.csv \
#                                     --save_dir ./

parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--fs_output_dir', dest='fs_output_dir',                      
                    help='path to fs_output_dir with all the subjects')

parser.add_argument('--stat_file', dest='stat_file',                     
                    default='aparc.DKTatlas.stats',
                    help='name of a standard FS stat file')

parser.add_argument('--stat_measure', dest='stat_measure',  
                    default='average_thickness_mm',
                    help='path to bids_dir')                    

parser.add_argument('--UKBB_dkt_ct_fields', dest='UKBB_dkt_ct_fields',  
                    help='UKBB lookup table with fields ID and ROI names')

parser.add_argument('--save_dir', dest='save_dir',  
                    default='./',
                    help='path to save_dir')

args = parser.parse_args()



if __name__ == "__main__":
    # Read from csv
    fs_output_dir = args.fs_output_dir
    stat_file = args.stat_file
    stat_measure = args.stat_measure
    save_dir = args.save_dir
    UKBB_dkt_ct_fields = args.UKBB_dkt_ct_fields

    UKBB_dkt_ct_fields_df = pd.read_csv(UKBB_dkt_ct_fields)

    print(f"Starting to collate {stat_measure} in {fs_output_dir}")
    subject_dir_list = glob.glob(f"{fs_output_dir}sub*")
    subject_id_list = [os.path.basename(x) for x in subject_dir_list]

    print(f"Found {len(subject_id_list)} subjects")

    hemispheres = ["lh", "rh"]

    hemi_stat_measures_dict = {}
    for hemi in hemispheres:
        stat_measure_df = pd.DataFrame()
        for subject_id in subject_id_list:
            try:
                fs_stats_dir = f"{fs_output_dir}{subject_id}/stats/"
                stats = CorticalParcellationStats.read(f"{fs_stats_dir}{hemi}.{stat_file}").structural_measurements
                
                cols = ["subject_id"] + list(stats["structure_name"].values)
                vals = [subject_id] + list(stats[stat_measure].values)
                
                df = pd.DataFrame(columns=cols)
                df.loc[0] = vals
                stat_measure_df = pd.concat([stat_measure_df, df], axis=0)
            except:
                print(f"Error parsing data for {subject_id}")

        # replace columns names with ukbb field IDs
        field_df = UKBB_dkt_ct_fields_df[UKBB_dkt_ct_fields_df["hemi"]==hemi][["Field ID","roi"]]
        roi_field_id_dict = dict(zip(field_df["roi"], field_df["Field ID"]))
        stat_measure_df = stat_measure_df.rename(columns=roi_field_id_dict)
        
        hemi_stat_measures_dict[hemi] = stat_measure_df

    # merge left and right dfs
    stat_measure_LR_df = pd.merge(hemi_stat_measures_dict["lh"],hemi_stat_measures_dict["rh"], on="subject_id")

    save_file = f"{stat_file.split('.')[1]}_{stat_measure.rsplit('_',1)[0]}.csv"

    print(f"Saving stat measures here: {save_dir}/{save_file}")
    stat_measure_LR_df.to_csv(f"{save_dir}/{save_file}")


