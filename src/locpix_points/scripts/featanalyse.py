"""Feature analysis module

Module takes in the .parquet files and analyses features
"""

import os
import yaml
import argparse
import json
import polars as pl
import time
import seaborn as sns
import matplotlib.pyplot as plt

def main(argv=None):
    # parse arugments
    parser = argparse.ArgumentParser(description="Analyse features")

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
        loc_files = os.listdir(os.path.join(project_directory, "preprocessed/featextract/locs"))
    except FileNotFoundError:
        raise ValueError("There should be some loc files to open")
    
    try:
        cluster_files = os.listdir(os.path.join(project_directory, "preprocessed/featextract/clusters"))
    except FileNotFoundError:
        raise ValueError("There should be some cluster files to open")
    
    assert loc_files == cluster_files

    # if output directory not present create it
    #output_loc_directory = os.path.join(
    #    project_directory, "preprocessed/featextract/locs"
    #)
    #output_cluster_directory = os.path.join(
    #    project_directory, "preprocessed/featextract/clusters"
    #)
    #folders = [output_loc_directory, output_cluster_directory]
    #for folder in folders:
    #    if not os.path.exists(folder):
    #        print("Making folder")
    #        os.makedirs(folder)

    # aggregate cluster features into collated df
    dfs = []

    for index, file in enumerate(loc_files):
        loc_path = os.path.join(project_directory, f"preprocessed/featextract/locs/{file}")
        cluster_path = os.path.join(project_directory, f"preprocessed/featextract/clusters/{file}")

        cluster_df = pl.read_parquet(cluster_path)
        if file.startswith('cancer'):
            cluster_df = cluster_df.with_columns(pl.lit('cancer').alias("type"))
        elif file.startswith('normal'):
            cluster_df = cluster_df.with_columns(pl.lit('normal').alias("type"))
        cluster_df = cluster_df.with_columns(pl.lit(f'{file}').alias("file_name"))
        dfs.append(cluster_df)

    # aggregate dfs into one big df
    df = pl.concat(dfs)

    df = df.to_pandas()

    print(df)

    
    sns.boxplot(data=df, x="RGyration", y="type")
    plt.show()
    sns.boxplot(data=df, x="linearity", y="type")
    plt.show()
    sns.boxplot(data=df, x="planarity", y="type")
    plt.show()
    sns.boxplot(data=df, x="length_pca", y="type")
    plt.show()
    sns.boxplot(data=df, x="area_pca", y="type")
    plt.show()
    sns.boxplot(data=df, x="perimeter", y="type")
    plt.show()
    sns.boxplot(data=df, x="area_convex_hull", y="type")
    plt.show()
    sns.boxplot(data=df, x="length_convex_hull", y="type")
    plt.show()
    # sns.pairplot(df, hue='type')
    # plt.show()

    # save yaml file
    yaml_save_loc = os.path.join(project_directory, "featextract.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
