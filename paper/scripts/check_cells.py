## Check locs per cell 
# 
# 1. Assert membrane and non-membrane have 5+ localisations

import os
import pyarrow.parquet as pq
import pyarrow.compute as pc
import polars as pl

input_folder = f"cells/gt_label"

files = os.listdir(input_folder)

# Localisation threshold
loc_threshold_global = 0 # threshold for localisations applied to each cell
loc_threshold_ind = 5 # threshold for localisations applied indepenedlty to membrane and non-membrane 

for file in files:
    # load memb and cell item
    df = pq.read_table(os.path.join(input_folder, file))
    if len(df) < loc_threshold_global:
       raise ValueError(f"Cell {file} has insufficient localisations per cell")
    else:
        # check membrane and non membrane
        non_memb_table = df.filter(pc.field("memb_label") == 0.0) 
        memb_table = df.filter(pc.field("memb_label") == 1.0) 
        
        if len(non_memb_table) < loc_threshold_ind or len(memb_table) < loc_threshold_ind:
            raise ValueError(f"Cell {file} has insufficient localisations per membrane/interior")
