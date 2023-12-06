"""Feature analysis module

Module takes in the .parquet files and analyses features
"""

import argparse
import json
import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl
import pyarrow.parquet as pq
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If no files present to open"""

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

    # parser.add_argument(
    #    "-c",
    #    "--config",
    #    action="store",
    #    type=str,
    #    help="the location of the .yaml configuaration file\
    #                         for processing",
    #    required=True,
    # )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # load config
    # with open(args.config, "r") as ymlfile:
    #    config = yaml.safe_load(ymlfile)

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
        loc_files = os.listdir(
            os.path.join(project_directory, "preprocessed/featextract/locs")
        )
    except FileNotFoundError:
        raise ValueError("There should be some loc files to open")

    try:
        cluster_files = os.listdir(
            os.path.join(project_directory, "preprocessed/featextract/clusters")
        )
    except FileNotFoundError:
        raise ValueError("There should be some cluster files to open")

    assert loc_files == cluster_files

    # if output directory not present create it
    # output_loc_directory = os.path.join(
    #    project_directory, "preprocessed/featextract/locs"
    # )
    # output_cluster_directory = os.path.join(
    #    project_directory, "preprocessed/featextract/clusters"
    # )
    # folders = [output_loc_directory, output_cluster_directory]
    # for folder in folders:
    #    if not os.path.exists(folder):
    #        print("Making folder")
    #        os.makedirs(folder)

    # aggregate cluster features into collated df
    dfs = []

    for index, file in enumerate(loc_files):
        # loc_path = os.path.join(
        #   project_directory, f"preprocessed/featextract/locs/{file}"
        # )
        cluster_path = os.path.join(
            project_directory, f"preprocessed/featextract/clusters/{file}"
        )

        cluster_df = pq.read_table(cluster_path)

        # extract metadata
        gt_label_map = json.loads(
            cluster_df.schema.metadata[b"gt_label_map"].decode("utf-8")
        )
        gt_label_map = {int(key): value for key, value in gt_label_map.items()}
        gt_label = cluster_df.schema.metadata[b"gt_label"]
        gt_label = int(gt_label)
        label = gt_label_map[gt_label]

        # convert to polars
        cluster_df = pl.from_arrow(cluster_df)
        cluster_df = cluster_df.with_columns(pl.lit(label).alias("type"))
        cluster_df = cluster_df.with_columns(pl.lit(f"{file}").alias("file_name"))
        dfs.append(cluster_df)

    # aggregate dfs into one big df
    df = pl.concat(dfs)
    df = df.to_pandas()

    sns.lineplot(data=df, x="length_pca", y="length_convex_hull")
    plt.show()
    sns.lineplot(data=df, x="area_pca", y="area_convex_hull")
    plt.show()

    sns.boxplot(data=df, x="count", y="type")
    plt.show()
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
    sns.boxplot(data=df, x="density_pca", y="type")
    plt.show()
    sns.boxplot(data=df, x="density_convex_hull", y="type")
    plt.show()

    input("stop")

    reducer = umap.UMAP()
    data = df[
        [
            "count",
            "RGyration",
            "linearity",
            "planarity",
            "length_pca",
            "area_pca",
            "perimeter",
            "area_convex_hull",
            "length_convex_hull",
            "density_pca",
            "density_convex_hull",
        ]
    ].values
    scaled_data = StandardScaler().fit_transform(data)
    embedding = reducer.fit_transform(scaled_data)

    # Plot UMAP - normal vs cancer
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in df.type.map({"normal": 0, "cancer": 1})],
        label=[x for x in df.type.map({"normal": 0, "cancer": 1})],
    )
    normal_patch = mpatches.Patch(color=sns.color_palette()[0], label="Normal")
    cancer_patch = mpatches.Patch(color=sns.color_palette()[1], label="Cancer")
    plt.legend(handles=[normal_patch, cancer_patch])
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection of the dataset", fontsize=24)
    plt.show()

    # Plot UMAP patients
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[
            sns.color_palette()[x]
            for x in df.file_name.map(
                {
                    "cancer_0.parquet": 0,
                    "cancer_1.parquet": 1,
                    "cancer_2.parquet": 2,
                    "normal_0.parquet": 3,
                    "normal_1.parquet": 4,
                    "normal_2.parquet": 5,
                }
            )
        ],
        label=[
            x
            for x in df.type.map(
                {
                    "cancer_0.parquet": 0,
                    "cancer_1.parquet": 1,
                    "cancer_2.parquet": 2,
                    "normal_0.parquet": 3,
                    "normal_1.parquet": 4,
                    "normal_2.parquet": 5,
                }
            )
        ],
    )
    # lgened
    cancer_patch_0 = mpatches.Patch(color=sns.color_palette()[0], label="Cancer 0")
    cancer_patch_1 = mpatches.Patch(color=sns.color_palette()[1], label="Cancer 1")
    cancer_patch_2 = mpatches.Patch(color=sns.color_palette()[2], label="Cancer 2")
    normal_patch_0 = mpatches.Patch(color=sns.color_palette()[3], label="Normal 0")
    normal_patch_1 = mpatches.Patch(color=sns.color_palette()[4], label="Normal 1")
    normal_patch_2 = mpatches.Patch(color=sns.color_palette()[5], label="Normal 2")
    plt.legend(
        handles=[
            cancer_patch_0,
            cancer_patch_1,
            cancer_patch_2,
            normal_patch_0,
            normal_patch_1,
            normal_patch_2,
        ]
    )
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection of the dataset", fontsize=24)
    plt.show()

    # save yaml file
    # yaml_save_loc = os.path.join(project_directory, "featextract.yaml")
    # with open(yaml_save_loc, "w") as outfile:
    #    yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
