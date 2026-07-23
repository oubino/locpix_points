"""Annotate module

Take in items, convert to histograms, annotate,
visualise histo mask, save the exported annotation .parquet
"""

import argparse
import json
import numpy as np
import os
import yaml

from locpix_points.preprocessing import datastruc


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If no files present to open or dimensions not 2 or 3
        NotImplementedError: If function not implemented yet
    """

    # parse arugments
    parser = argparse.ArgumentParser(description="Annotate the data")

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
                             for annotation",
        required=True,
    )

    parser.add_argument(
        "-r",
        "--relabel",
        action="store_true",
        default=False,
        help="If true will relabel and assume labels are present (default = False)",
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
    output_datastructure_directory = os.path.join(project_directory, "preprocessed/gt_label")
    if not os.path.exists(output_datastructure_directory):
        print("Making folder")
        os.makedirs(output_datastructure_directory)

    if args.napari:

        # if output directory not present create it
        output_labels_directory = os.path.join(project_directory, "preprocessed/labels")
        if not os.path.exists(output_labels_directory):
            print("Making folder")
            os.makedirs(output_labels_directory)
        
        # if output directory not present create it
        output_markers_directory = os.path.join(project_directory, "preprocessed/markers")
        if not os.path.exists(output_markers_directory):
            print("Making folder")
            os.makedirs(output_markers_directory)

    for file in files:
        item = datastruc.item(None, None, None, None, None)
        item.load_from_parquet(
            os.path.join(project_directory, "preprocessed/no_gt_label", file)
        )

        # check if file already present and annotated
        # note assumptions
        # 1. assumes name convention of save_to_parquet is
        # os.path.join(save_folder, self.name + '.parquet')
        parquet_save_loc = os.path.join(output_datastructure_directory, item.name + ".parquet")
        if os.path.exists(parquet_save_loc) and not args.relabel:
            print(f"Skipping file as already present: {parquet_save_loc}")
            continue

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
            )

            labels_loc = os.path.join(output_labels_directory, item.name + ".npy")
            markers_loc = os.path.join(output_markers_directory, item.name + ".npy")

            # manual segment
            labels, markers = item.manual_segment_per_loc(
                relabel=args.relabel,
                labels_loc=labels_loc,
                markers_loc=markers_loc,
            )

            # save df to parquet
            item.gt_label_scope = "loc"
            item.gt_label = None

            # save labels
            np.save(labels_loc, labels)

            # save markers
            np.save(markers_loc, markers)


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
            output_datastructure_directory,
            drop_zero_label=config["drop_zero_label"],
            overwrite=args.relabel,
        )

    # save gt label map to metadata
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        metadata["gt_label_map"] = config["gt_label_map"]
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)


if __name__ == "__main__":
    main()
