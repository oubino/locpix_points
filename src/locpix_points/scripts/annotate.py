"""Annotate module

Take in items, convert to histograms, annotate,
visualise histo mask, save the exported annotation .parquet
"""

import yaml
import os
from locpix_points.preprocessing import datastruc
import polars as pl
import argparse


def main():

    # parse arugments
    parser = argparse.ArgumentParser(description="Annotate the data")

    parser.add_argument(
        "-i",
        "--project_direcotry",
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
                             for annotation",
        required=True,
    )

    args = parser.parse_args()

    project_directory = args.project_directory

    # load yaml
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(
            os.path.join(project_directory, "preprocessed/not_annotated")
        )
    except FileNotFoundError:
        raise ValueError("There should be some preprocessed files to open")

    # if output directory not present create it
    output_directory = os.path.join(project_directory, "preprocessed/annotated")
    if not os.path.exists(output_directory):
        print("Making folder")
        os.makedirs(output_directory)

    if config["dim"] == 2:
        histo_size = (config["x_bins"], config["y_bins"])
    elif config["dim"] == 3:
        histo_size = (config["x_bins"], config["y_bins"], config["z_bins"])
    else:
        raise ValueError("Dim should be 2 or 3")

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(
            os.path.join(project_directory, "preprocessed/not_annotated", file)
        )

        # coord2histo
        item.coord_2_histo(
            histo_size,
            plot=config["plot"],
            vis_interpolation=config["vis_interpolation"],
        )

        # manual segment
        item.manual_segment_per_loc()

        # save df to parquet 
        item.gt_label_scope = 'loc'
        item.gt_label = None
        item.gt_label_map = config['gt_label_map']
        item.save_to_parquet(
            output_directory,
            drop_zero_label=config["drop_zero_label"],
        )


if __name__ == "__main__":
    main()
