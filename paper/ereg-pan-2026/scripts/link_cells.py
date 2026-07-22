## CANCER_OUTCOMES_PAN

# Label the data with the patient outcomes, grouping into 
#   response (r = 1,2), and 
#   no response (r = 4,6,9)

# where...
#   1: 'Complete response'
#   2:'Partial response'
#   4:'Radiological progression'
#   6:"Clinical progression"
#   9:"Death"

# Further we only look at 
#   cancer FOVs, 
#   patients assigned to the panitumumab arm, 
#   patients who are WT

import os
import polars as pl
import json
import pyarrow.parquet as pq

input_linked_files = 'config/linked_files.csv'
input_folder = 'cells/raw'
output_folder = 'cells/gt_label'
gt_label_map = {0: 'no_response', 1: "any_response"}
gt_label_map = json.dumps(gt_label_map).encode("utf-8")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

## Load in linked files sheet
linked_df = pl.read_csv(input_linked_files)

# ## Link cell data and save

# link with outcomes
files = os.listdir(input_folder)

any_response_total = 0
no_response_total = 0

for file in files:

    df = pq.read_table(f'{input_folder}/{file}')

    file_name = file

    # fov name
    pid = file_name[0:file_name.index("_cell")]

    if pid in linked_df["patient"]:

        # link with outcome
        response = linked_df.filter(
            pl.col("patient") == pid
        )[["response"]].item()
        response = int(response)

        if response == 0:
            no_response_total +=1 
        else:
            any_response_total += 1

        # remove gt_label column
        df = df.drop("gt_label")

        # remove old metadata
        old_metadata = df.schema.metadata
        old_metadata.pop(b'gt_label')
        old_metadata.pop(b'gt_label_map')
        old_metadata.pop(b'gt_label_scope')

        new_metadata = {
            "gt_label": str(response),
            "gt_label_map": gt_label_map,
            "gt_label_scope": "fov",
        }

        ## merge existing with new meta data and save
        merged_metadata = {**new_metadata, **(old_metadata or {})}
        df = df.replace_schema_metadata(merged_metadata)
        save_loc =  f'{output_folder}/{file}'
        pq.write_table(df, save_loc)
    
    else:
        raise ValueError(f"{pid} should be present in file list")

print("------------")
print("No response: ", no_response_total)
print("Any response: ", any_response_total)