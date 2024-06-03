"""Preprocessing module

Module takes in the .csv files and processes saving the datastructures
"""

import argparse
import json
import os
import socket
import time

import yaml

from locpix_points.preprocessing import functions


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If try to preprocess but already files there

    Returns:
        1: Returns 1 in the case that files are already preprocessed"""

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
    parser.add_argument(
        "-f",
        "--preprocessed_folder",
        action="store",
        type=str,
        help="the location of the preprocessed folder relative to the project directory e.g."
        " preprocessed/train",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory
    input_folder = args.input

    # create preprocessed directory
    if args.preprocessed_folder is None:
        output_folder = os.path.join(project_directory, "preprocessed")
    else:
        output_folder = os.path.join(project_directory, args.preprocessed_folder)
    # If already preprocessed e.g. copying from other folder then don't need to run and should skip
    if os.path.exists(output_folder):
        return 1
    else:
        os.makedirs(output_folder)

    # save to metadata
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
    gt_label_map = None
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
                output_directory = os.path.join(output_folder, "gt_label")
                os.makedirs(output_directory)
            else:
                output_directory = os.path.join(output_folder, "no_gt_label")
                os.makedirs(output_directory)

        item.save_to_parquet(
            output_directory,
            drop_zero_label=False,
            drop_pixel_col=config["drop_pixel_col"],
        )

        if gt_label_map is None:
            gt_label_map = item.gt_label_map
        else:
            assert gt_label_map == item.gt_label_map

    # save gt label map to metadata
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        metadata["gt_label_map"] = gt_label_map
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # save yaml file
    yaml_save_loc = os.path.join(project_directory, "preprocess.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
