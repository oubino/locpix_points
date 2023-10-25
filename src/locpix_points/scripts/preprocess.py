"""Preprocessing module

Module takes in the .csv files and processes saving the datastructures
"""

import os
import yaml
from locpix_points.preprocessing import functions
import argparse
import socket
import json
import time


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
        """Save the metadata as a .csv

        Args:
            path (string) : Path to save to"""

        with open(path, "w") as outfile:
            json.dump(self.metadata, outfile)

    def load(self, path):
        """Load the metadata

        Args:
            path (string) : Path to load from"""

        self.metadata = json.load(path)


def main():
    # parse arugments
    parser = argparse.ArgumentParser(
        description="Preprocess the data for\
        further processing."
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
        help="the location of the .yaml configuaration file\
                             for preprocessing",
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

    project_directory = args.project_directory
    input_folder = args.input

    # create project directory
    output_folder = os.path.join(project_directory, "preprocessed")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # initialise metadata and save
        metadata = project_info(
            time.asctime(time.gmtime(time.time())), project_directory
        )
        metadata.save(os.path.join(project_directory, "metadata.json"))

    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # if all is specified then consider all files otherwise consider specified files
    if config["include_files"] == "all":
        include_files = os.listdir(args.input)
        include_files = [os.path.splitext(item)[0] for item in include_files]
    else:
        include_files = config["include_files"]

    # check with user
    print("List of files which will be processed")
    files = [os.path.join(input_folder, f"{file}.parquet") for file in include_files]
    # check file not already present
    for file in files:
        file_name = os.path.basename(file)
        output_path = os.path.join(output_folder, f"{file_name}")
        if os.path.exists(output_path):
            raise ValueError("Can't preprocess as output file already exists")
    print(files)
    # check = input("If you are happy with these parquets type YES: ")
    # if check != "YES":
    #    exit()

    # go through files -> convert to datastructure -> save
    for index, file in enumerate(files):
        item = functions.file_to_datastruc(
            file,
            config["dim"],
            config["channel_col"],
            config["frame_col"],
            config["x_col"],
            config["y_col"],
            config["z_col"],
            config["channel_choice"],
            config["channel_label"],
            config["gt_label_scope"],
            config["gt_label_loc"],
            config["features"],
        )

        # if succesfully get to here on first occasion create folder for data
        if index == 0:
            if config["gt_label_scope"] is not None:
                output_directory = os.path.join(
                    project_directory, "preprocessed/gt_label"
                )
                os.makedirs(output_directory)
            else:
                output_directory = os.path.join(
                    project_directory, "preprocessed/no_gt_label"
                )
                os.makedirs(output_directory)

        item.save_to_parquet(
            output_directory,
            drop_zero_label=False,
            drop_pixel_col=config["drop_pixel_col"],
        )

    # save yaml file
    yaml_save_loc = os.path.join(
        project_directory, f"preprocess_{os.path.basename(input_folder)}.yaml"
    )
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
