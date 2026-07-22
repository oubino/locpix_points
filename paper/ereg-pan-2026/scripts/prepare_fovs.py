#!/usr/bin/env python
"""Preprocessing module

Module takes in the .csv files and processes saving the datastructures
"""

import argparse
import json
import os
import polars as pl
import socket
import time
import yaml

from locpix_points.preprocessing import datastruc, functions

class project_info:
    """Project information metadata

    Attributes:
        metadata (dictionary) : Python dictionary containing
            the metadata"""

    def __init__(self, time, name):
        """Initialises metadata with args

        Args:
            time (string) : Time of project initialisation
            name (string) : Name of the project"""

        # dictionary
        self.metadata = {
            "machine": socket.gethostname(),
            "name": name,
            "init_time": time,
        }

    def save(self, path):
        """Save the metadata

        Args:
            path (string) : Path to save to"""

        with open(path, "w") as outfile:
            json.dump(self.metadata, outfile)

    def load(self, path):
        """Load the metadata

        Args:
            path (string) : Path to load from"""

        self.metadata = json.load(path)

def load_csv(
        input_file,
        dim,
        channel_col,
        frame_col,
        x_col,
        y_col,
        channel_choice,
        channel_label,
    ):
    
    df = pl.read_csv(
        input_file, 
        columns=[channel_col, frame_col, x_col, y_col]
    )
        
    df = df.rename(
        {channel_col: "channel", frame_col: "frame", x_col: "x", y_col: "y"}
    )

    df = df.filter(pl.col("channel").is_in(channel_choice))

    # Get name of file - assumes last part of input file name
    name = os.path.basename(os.path.normpath(input_file)).removesuffix(".csv")

    return datastruc.item(
        name,
        df,
        dim,
        channel_choice,
        channel_label,
    )

def main():

    # load path of .csv or .parquet
    parser = argparse.ArgumentParser(
        description="Prepare the FOVs."
    )

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        type=str,
        help="path for the input data folder",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="location of the .yaml configuaration file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
        required=True,
    )

    args = parser.parse_args()

    input_path = args.input
    project_folder = args.project_directory
    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # if output directory not present create it
    output_folder = os.path.join(project_folder, "preprocessed/no_gt_label")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # initialise metadata and save
        metadata = project_info(time.asctime(time.gmtime(time.time())), project_folder)
        metadata.save(os.path.join(project_folder, "metadata.json"))

    # if all is specified then consider all files otherwise consider specified files
    include_files = os.listdir(args.input)
    include_files = [os.path.splitext(item)[0] for item in include_files]

    files = [os.path.join(input_path, f"{file}.csv") for file in include_files]
    # check file not already present
    for file in files:
        file_name = os.path.basename(file)
        output_path = os.path.join(
            output_folder, f"{file_name.replace('.csv', '.parquet')}"
        )
        if os.path.exists(output_path):
            raise ValueError("Can't preprocess as output file already exists")

    # go through files -> convert to datastructure -> save
    for file in files:
        item = load_csv(
            file,
            config["dim"],
            config["channel_col"],
            config["frame_col"],
            config["x_col"],
            config["y_col"],
            config["channel_choice"],
            config["channel_label"],
        )
        # have to not drop zero label
        # as no gt_label yet
        item.save_to_parquet(
            output_folder,
        )

if __name__ == "__main__":
    main()