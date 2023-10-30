"""Feature extraction module

Module takes in the .parquet files and extracts features
"""

import os
import yaml
from locpix_points.preprocessing import functions
import argparse
import json
import time
from locpix_points.data_loading import datastruc
from locpix_points.preprocessing import featextract


def main(argv=None):
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

    args = parser.parse_args(argv)

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

    # list items
    try:
        files = os.listdir(os.path.join(project_directory, "preprocessed/gt_label"))
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    output_loc_directory = os.path.join(
        project_directory, "preprocessed/featextract/locs"
    )
    output_cluster_directory = os.path.join(
        project_directory, "preprocessed/featextract/clusters"
    )
    folders = [output_loc_directory, output_cluster_directory]
    for folder in folders:
        if not os.path.exists(folder):
            print("Making folder")
            os.makedirs(folder)

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(project_directory, file))

        # clustering (clusterID)
        df = featextract.cluster_data(
            item.df,
            eps=config["clustering"]["eps"],
            minpts=config["clustering"]["minpts"],
            x_col="x",
            y_col="y",
        )

        # basic features (com cluster, locs per cluster, radius of gyration)
        basic_cluster_df = featextract.basic_cluster_feats(df)

        # pca on cluster (linearity, circularity see DIMENSIONALITY BASED SCALE SELECTION IN 3D LIDAR POINT CLOUDS)
        pca_cluster_df = featextract.pca_cluster(df)

        # convex hull (perimeter, area, length)
        convex_hull_cluster_df = featextract.convex_hull_cluster(df)

        # merge cluster df
        cluster_df = basic_cluster_df.join(
            pca_cluster_df, on="clusterID", how="inner"
        ) 
        print('post first join ', len(cluster_df))
        cluster_df = cluster_df.join(
            convex_hull_cluster_df, on="clusterID", how="inner"
        )
        print('post second join ', len(cluster_df))
        print('basic/pca/convex hull length')
        print(len(basic_cluster_df))
        print(len(pca_cluster_df))
        print(len(convex_hull_cluster_df))

        raise ValueError("Need to check this join is okay")

        # cluster density do this here
        cluster_df.with_columns(pl.col("count") / pl.col("area")).alias("density")

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

        raise ValueError("Check gt label map and gt label fov")

    # save yaml file
    yaml_save_loc = os.path.join(project_directory, "featextract.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
