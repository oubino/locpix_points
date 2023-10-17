"""Feature extraction module

Module takes in the .parquet files and extracts features
"""

import os
import yaml
from locpix_points.preprocessing import functions
import argparse
import json
import time

def main():

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Extract features"
    )

    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="location of the project directory",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for processing",
        required=True,
    )

    args = parser.parse_args()

    project_directory = args.project_directory

    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        file = os.path.basename(__file__)
        if file not in metadata:
            metadata[file] = time.asctime(time.gmtime(time.time()))
        else:
            print("Overwriting metadata...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # for each fov
    
    # extract features

    # clustering

    # pca on cluster
    

    # convex hull

    # cluster skew, num locs, density, radius of gyration, length, distance birth/death

    # convex hull - area, perimeter

    # cluster df save to item separately to reduce redundancy

    # circularity: null
    
    # save yaml file
    yaml_save_loc = os.path.join(project_directory, "process.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)

if __name__ == "__main__":
    main()
