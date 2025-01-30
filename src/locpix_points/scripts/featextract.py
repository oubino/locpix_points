"""Feature extraction module

Module takes in the .parquet files and extracts features
"""

import argparse
import json
import os
import time
import warnings

import polars as pl
import yaml
from dask.distributed import Client

from locpix_points.preprocessing import datastruc, featextract


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If no files present to open OR
            unsupported clustering method"""

    # parse arugments
    parser = argparse.ArgumentParser(description="Extract features")

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

    parser.add_argument(
        "-f",
        "--preprocessed_folder",
        action="store",
        type=str,
        help="the location of the preprocessed folder relative to the project directory",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    if args.preprocessed_folder is None:
        preprocessed_folder = os.path.join(project_directory, "preprocessed")
    else:
        preprocessed_folder = os.path.join(project_directory, args.preprocessed_folder)

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

    # list items
    try:
        files = os.listdir(os.path.join(preprocessed_folder, "gt_label"))
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    output_loc_directory = os.path.join(preprocessed_folder, "featextract/locs")
    output_cluster_directory = os.path.join(preprocessed_folder, "featextract/clusters")
    folders = [output_loc_directory, output_cluster_directory]
    for folder in folders:
        if not os.path.exists(folder):
            print("Making folder")
            os.makedirs(folder)

    # start client for dask
    _ = Client()

    # remove from files the ones that have already had feat extracted
    loc_files = os.listdir(output_loc_directory)
    cluster_files = os.listdir(output_cluster_directory)
    completed_files = []
    for file in files:
        if file in loc_files and file in cluster_files:
            completed_files.append(file)
    files = [file for file in files if file not in completed_files]
    for index, file in enumerate(files):
        print("file", file)
        item = datastruc.item(None, None, None, None, None)
        item.load_from_parquet(os.path.join(preprocessed_folder, f"gt_label/{file}"))

        # clustering (clusterID)
        if "dbscan" in config.keys():
            df = featextract.cluster_data(
                item.df,
                eps=config["dbscan"]["eps"],
                minpts=config["dbscan"]["minpts"],
                x_col="x",
                y_col="y",
                method="dbscan",
            )
        elif "kmeans" in config.keys():
            df = featextract.cluster_data(
                item.df,
                n_clusters=config["kmeans"]["n_clusters"],
                x_col="x",
                y_col="y",
                method="kmeans",
            )
        else:
            raise ValueError("Only support dbscan or kmeans clustering")

        # drop locs not clustered
        # warnings.warn("Drop all unclustered points")
        df = df.filter(pl.col("clusterID") != -1)

        # drop locs with only 2 loc
        # warnings.warn(
        #    "Dropping all clusters with 2 or fewer locs - otherwise convex hull/PCA fail"
        # )
        small_clusters = df.group_by("clusterID").count().filter(pl.col("count") < 3)
        df = df.filter(~pl.col("clusterID").is_in(small_clusters["clusterID"]))

        # remap the clusterIDs
        unique_clusters = list(df["clusterID"].unique(maintain_order=True))
        map = {value: i for i, value in enumerate(unique_clusters)}
        df = df.with_columns(pl.col("clusterID").replace(map).alias("clusterID"))

        # warnings.warn("If no clusters then rest will fail")

        # basic features (com cluster, locs per cluster, radius of gyration)
        basic_cluster_df = featextract.basic_cluster_feats(df)

        # pca on cluster (linearity, circularity see DIMENSIONALITY BASED SCALE SELECTION IN 3D LIDAR POINT CLOUDS)
        pca_cluster_df = featextract.pca_cluster(df)

        # convex hull (perimeter, area, length)
        convex_hull_cluster_df = featextract.convex_hull_cluster(df)

        # merge cluster df
        cluster_df = basic_cluster_df.join(pca_cluster_df, on="clusterID", how="inner")
        cluster_df = cluster_df.join(
            convex_hull_cluster_df, on="clusterID", how="inner"
        )

        # don't allow fewer than 3 clusters
        num_clusters = cluster_df["clusterID"].max()
        if num_clusters < 3:
            raise ValueError("2 or fewer clusters")

        # cluster density do this here
        cluster_df = cluster_df.with_columns(
            (pl.col("count") / pl.col("area_convex_hull")).alias("density_convex_hull")
        )
        cluster_df = cluster_df.with_columns(
            (pl.col("count") / pl.col("area_pca")).alias("density_pca")
        )

        # identify superclusters
        if "superclusters" in config.keys():
            raise ValueError(
                "Currently we calculate superclusters as an add-on via the supercluster script"
            )
            if config["superclusters"]:
                cluster_df = featextract.super_cluster(
                    cluster_df,
                    k=15,
                )

        # save locs dataframe
        item.df = df
        item.save_to_parquet(
            output_loc_directory,
            drop_zero_label=False,
            drop_pixel_col=False,
        )

        # save clusters dataframe
        item.df = cluster_df
        item.save_to_parquet(
            output_cluster_directory,
            drop_zero_label=False,
            drop_pixel_col=False,
        )

    # save yaml file
    yaml_save_loc = os.path.join(project_directory, "featextract.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
