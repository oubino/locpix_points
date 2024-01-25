"""Annotate module

Take in items, convert to histograms, annotate,
visualise histo mask, save the exported annotation .parquet
"""

import argparse
import os
import yaml

from locpix_points.preprocessing import datastruc


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If no files present to open or dimensions not 2 or 3"""
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

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-n",
        "--napari",
        action="store_true",
        help="if specified then we use napari to annotate each localisation",
    )
    group.add_argument(
        "-s",
        "--scope",
        choices=["fov", "loc"],
        help="if fov then label is per fov, if loc then label is per loc",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # load yaml
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(os.path.join(project_directory, "preprocessed/no_gt_label"))
    except FileNotFoundError:
        raise ValueError("There should be some preprocessed files to open")

    # if output directory not present create it
    output_directory = os.path.join(project_directory, "preprocessed/gt_label")
    if not os.path.exists(output_directory):
        print("Making folder")
        os.makedirs(output_directory)

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(
            os.path.join(project_directory, "preprocessed/no_gt_label", file)
        )

        if args.napari:
            if config["napari"]["dim"] == 2:
                histo_size = (config["napari"]["x_bins"], config["napari"]["y_bins"])
            elif config["dim"] == 3:
                histo_size = (
                    config["napari"]["x_bins"],
                    config["napari"]["y_bins"],
                    config["napari"]["z_bins"],
                )
            else:
                raise ValueError("Dim should be 2 or 3")

            # coord2histo
            item.coord_2_histo(
                histo_size,
                plot=config["napari"]["plot"],
                vis_interpolation=config["napari"]["vis_interpolation"],
            )

            # manual segment
            item.manual_segment_per_loc()

            # save df to parquet
            item.gt_label_scope = "loc"
            item.gt_label = None

        else:
            if args.scope == "fov":
                item.gt_label_scope = "fov"
                raise NotImplementedError(
                    "User needs to implement their own annotation"
                )
                # Need to assign a label to the fov
                # Below is an example
                # if file.startswith('positive_case'):
                #     label = 1
                # else:
                #     label = 0
                #
                # item.gt_label = label

            elif args.scope == "loc":
                item.gt_label_scope = "loc"
                item.gt_label = None
                raise NotImplementedError("This is not implemented yet")
                # Need to assign a label to each localisation
                # Below are examples
                # 1) Add label 0 to all localisations
                # item.df.with_columns(
                #    pl.lit(0).alias("gt_label")
                # )
                # 2) Add label 2* value in column called x
                # item.df.with_columns(
                #    (pl.col("x") * 2).alias("gt_label")
                # )

            else:
                raise ValueError("Scope should be fov or loc")

        item.gt_label_map = config["gt_label_map"]
        item.save_to_parquet(
            output_directory,
            drop_zero_label=config["drop_zero_label"],
        )


if __name__ == "__main__":
    main()
