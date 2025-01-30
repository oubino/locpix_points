"""Process recipe

Recipe :
    1. Create dataset
    2. Process dataset - pre-transform and save to .pt
"""
import argparse
import ast
import json
import os
import random
import time
import warnings
from functools import partial
from collections import Counter

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import yaml

from locpix_points.data_loading import datastruc

# import torch_geometric.transforms as T


def pre_filter(data, inclusion_list=[]):
    """Takes in data item and returns whether
    it should be included in final dataset
    i.e. 1 - yes ; 0 - no

    Args:
        data (torch.geometric.data) : The pytorch
            geometric dataitem part of the dataset
        inclusion_list (list) : List of names
            indicating which data should be included

    Returns:
        0/1: 0 if file shouldn't be included in final dataset and 1
            if it should"""

    if data.name in inclusion_list:
        return 1
    else:
        return 0


def load_pre_filter(path):
    """Load in a pre-filter from previous run

    Args:
        path (string) : Path to the pre-filter.pt

    Returns:
        pre_filter (?) : Pre filter saved

    Raises:
        ValueError : If pre-filter is not correct type"""

    pre_filter = torch.load(path)
    if (
        pre_filter.startswith("functools.partial(<function>, inclusion_list=")
        and pre_filter[-1] == ")"
    ):
        pre_filter = pre_filter.removeprefix(
            "functools.partial(<function>, inclusion_list="
        )
        pre_filter = pre_filter[:-1:]
        return ast.literal_eval(pre_filter)
    else:
        raise ValueError("Unknown pre-filter type")


def minmax(config, feat_str, file_directory, train_list):
    """Calculate minimum and maximum values of the features
    in the file

    Args:
        config (dict): Configuration for processing
        feat_str (str): loc_feat or cluster_feat depending on if
            calculating for the localisations or clusters
        file_directory (str): Directory containing the files
        train_list (list): List of files to load in

    Returns:
        min_vals (dict): Dictioanry containing the minimum values for the
            features
        max_vals (dict): Dictioanry containing the maximum values for the
            features

    Raises:
        NotImplementedError: If the features in the config file
            are not correct
    """
    if type(config[feat_str]) is list and len(config[feat_str]) != 0:
        for index, file in enumerate(train_list):
            df = pl.read_parquet(os.path.join(file_directory, file + ".parquet"))
            min_df = df.select(pl.col(config[feat_str]).min())
            max_df = df.select(pl.col(config[feat_str]).max())
            if index == 0:
                min_vals = min_df.to_numpy()[0]
                max_vals = max_df.to_numpy()[0]
            else:
                min_vals = np.min((min_vals, min_df.to_numpy()[0]), axis=0)
                max_vals = np.max((max_vals, max_df.to_numpy()[0]), axis=0)
    elif type(config[feat_str]) is list and len(config[feat_str]) == 0:
        return None, None
    elif config[feat_str] is None:
        return None, None
    else:
        raise NotImplementedError

    min_vals = dict(zip(config[feat_str], min_vals))
    max_vals = dict(zip(config[feat_str], max_vals))

    return min_vals, max_vals


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        NotImplementedError: If model type not recognised
        ValueError: temporary"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Preprocess the data for\
        further processing."
    )

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
        "-o",
        "--output_folder",
        action="store",
        type=str,
        help="location of the output folder, if not specified defaults\
                to project_directory/procseed",
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-r",
        "--split",
        action="store",
        type=str,
        default=None,
        help="if you want to copy the data split of another project then include this argument with\
              the location of the project folder",
    )

    group.add_argument(
        "-m",
        "--manual_split",
        type=str,
        nargs="+",
        action="append",
        default=None,
        help="list of lists, list[0]=train files, list[1] = val files, list[2] = test files",
    )

    group.add_argument(
        "-k",
        "--k_split",
        type=str,
        nargs="+",
        action="append",
        default=None,
        help="list of lists, list[0]=train files, list[1] = val files, list[2] = test files\
              has to be slightly different to manual split",
    )

    group.add_argument(
        "-f",
        "--final_test",
        type=str,
        nargs="+",
        action="append",
        default=None,
        help="list of paths, list[0] = path to train folder, list[1] = path to test folder"
        "each path is relative to the project folder",
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

    # Directory of the .parquet files have been
    # preprocessed by preprocessing module but
    # are raw with respect to pytorch/analysis
    if args.output_folder is not None:
        processed_dir_root = os.path.join(project_directory, args.output_folder)
    else:
        processed_dir_root = os.path.join(project_directory, "processed")

    # split into train/val/test using pre filter

    # split randomly
    if (
        args.split is None
        and args.manual_split is None
        and args.k_split is None
        and args.final_test is None
    ):
        file_list = os.listdir(os.path.join(project_directory, "preprocessed/gt_label"))
        file_list = [file.removesuffix(".parquet") for file in file_list]
        random.shuffle(file_list)
        # split into train/test/val
        train_length = int(len(file_list) * config["train_ratio"])
        test_length = int(len(file_list) * config["test_ratio"])
        val_length = len(file_list) - train_length - test_length
        train_list = file_list[0:train_length]
        val_list = file_list[train_length : train_length + val_length]
        test_list = file_list[train_length + val_length : len(file_list)]

    elif args.split is not None:
        warnings.warn(
            "Known omission is if pre-transform is done to dataset"
            "this is not currently also done to this dataset as well"
        )

        train_list_path = os.path.join(args.split, "processed/train/pre_filter.pt")
        val_list_path = os.path.join(args.split, "processed/val/pre_filter.pt")
        test_list_path = os.path.join(args.split, "processed/test/pre_filter.pt")

        train_list = load_pre_filter(train_list_path)
        val_list = load_pre_filter(val_list_path)
        test_list = load_pre_filter(test_list_path)

    elif args.manual_split is not None:
        train_list = args.manual_split[0]
        val_list = args.manual_split[1]
        test_list = args.manual_split[2]

    elif args.k_split is not None:
        train_list = ast.literal_eval(args.k_split[0][0])
        val_list = ast.literal_eval(args.k_split[1][0])
        test_list = ast.literal_eval(args.k_split[2][0])

    elif args.final_test is not None:
        # this will generate the same train/val split each time

        train_val_list = os.listdir(
            os.path.join(project_directory, args.final_test[0][0], "gt_label")
        )
        train_val_list = [file.removesuffix(".parquet") for file in train_val_list]
        test_list = os.listdir(
            os.path.join(project_directory, args.final_test[1][0], "gt_label")
        )
        test_list = [file.removesuffix(".parquet") for file in test_list]

        # check ratio
        assert config["train_ratio"] + config["val_ratio"] == 1.0

        # split in correct proportions for class
        classes = []
        for file in train_val_list:
            file = pq.read_table(
                os.path.join(
                    project_directory,
                    args.final_test[0][0],
                    "gt_label",
                    file + ".parquet",
                )
            )
            classes.append(int(file.schema.metadata[b"gt_label"]))
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=config["val_ratio"], random_state=0
        )
        x = np.zeros(len(classes))
        for i in sss.split(x, classes):
            train, val = i
        train_val_list = np.array(train_val_list)
        train_list = train_val_list[train.tolist()]
        val_list = train_val_list[val.tolist()]

        train_list_check = [x.split("_")[0] for x in train_list]
        val_list_check = [x.split("_")[0] for x in val_list]
        counter_train = Counter(train_list_check)
        counter_val = Counter(val_list_check)
        print(counter_train.keys())
        print(counter_train.values())
        print(counter_val.keys())
        print(counter_val.values())
        raise ValueError(
            "Need to check above method returns correct proportions of each class in validation set & sets are disjoint"
        )

    # bind arguments to functions
    train_pre_filter = partial(pre_filter, inclusion_list=train_list)
    val_pre_filter = partial(pre_filter, inclusion_list=val_list)
    test_pre_filter = partial(pre_filter, inclusion_list=test_list)

    # folders
    train_folder = os.path.join(processed_dir_root, "train")
    val_folder = os.path.join(processed_dir_root, "val")
    test_folder = os.path.join(processed_dir_root, "test")
    # if output directory not present create it
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # get correct input folders
    if args.final_test is None:
        input_folder_train = os.path.join(project_directory, "preprocessed")
        input_folder_val = input_folder_train
        input_folder_test = input_folder_train
    else:
        input_folder_train = os.path.join(project_directory, args.final_test[0][0])
        input_folder_val = input_folder_train
        input_folder_test = os.path.join(project_directory, args.final_test[1][0])

    # calculate min/max features on training data
    if config["model"] == "ClusterLoc":
        file_directory = os.path.join(input_folder_train, "featextract/locs")
        min_feat_locs, max_feat_locs = minmax(
            config, "loc_feat", file_directory, train_list
        )
        file_directory = os.path.join(input_folder_train, "featextract/clusters")
        min_feat_clusters, max_feat_clusters = minmax(
            config, "cluster_feat", file_directory, train_list
        )

        if "superclusters" in config.keys():
            input("check got here")
            superclusters = True
        else:
            superclusters = False

        print("Train set...")
        # create train dataset
        _ = datastruc.ClusterLocDataset(
            os.path.join(input_folder_train, "featextract/locs"),
            os.path.join(input_folder_train, "featextract/clusters"),
            train_folder,
            config["label_level"],
            train_pre_filter,
            config["save_on_gpu"],
            None,  # transform introduced in train script
            None,  # pre-transform
            config["loc_feat"],
            config["cluster_feat"],
            min_feat_locs,
            max_feat_locs,
            min_feat_clusters,
            max_feat_clusters,
            config["kneighboursclusters"],
            config["fov_x"],
            config["fov_y"],
            kneighbourslocs=config["kneighbourslocs"],
            superclusters=superclusters,
        )

        print("Val set...")
        # create val dataset
        _ = datastruc.ClusterLocDataset(
            os.path.join(input_folder_val, "featextract/locs"),
            os.path.join(input_folder_val, "featextract/clusters"),
            val_folder,
            config["label_level"],
            val_pre_filter,
            config["save_on_gpu"],
            None,  # transform
            None,  # pre-transform
            config["loc_feat"],
            config["cluster_feat"],
            min_feat_locs,
            max_feat_locs,
            min_feat_clusters,
            max_feat_clusters,
            config["kneighboursclusters"],
            config["fov_x"],
            config["fov_y"],
            kneighbourslocs=config["kneighbourslocs"],
            superclusters=superclusters,
        )

        print("Test set...")
        # create test dataset
        _ = datastruc.ClusterLocDataset(
            os.path.join(input_folder_test, "featextract/locs"),
            os.path.join(input_folder_test, "featextract/clusters"),
            test_folder,
            config["label_level"],
            test_pre_filter,
            config["save_on_gpu"],
            None,  # transform
            None,  # pre-transform
            config["loc_feat"],
            config["cluster_feat"],
            min_feat_locs,
            max_feat_locs,
            min_feat_clusters,
            max_feat_clusters,
            config["kneighboursclusters"],
            config["fov_x"],
            config["fov_y"],
            kneighbourslocs=config["kneighbourslocs"],
            superclusters=superclusters,
        )

        # save yaml file
        yaml_save_loc = os.path.join(project_directory, "process.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)

    elif config["model"] == "Loc":
        if os.path.exists(os.path.join(input_folder_train, "featextract/locs")):
            input_folder = "featextract/locs"
            min_feat, max_feat = minmax(config, "loc_feat", file_directory, train_list)
        else:
            input_folder = "gt_label"
            min_feat = None
            max_feat = None

        print("Train set...")
        # create train dataset
        _ = datastruc.LocDataset(
            os.path.join(input_folder_train, input_folder),
            train_folder,
            config["label_level"],
            train_pre_filter,
            config["save_on_gpu"],
            None,  # transform introduced in train script
            None,  # pre-transform
            config["feat"],
            min_feat,
            max_feat,
            config["fov_x"],
            config["fov_y"],
            kneighbours=config["kneighbours"],
        )

        print("Val set...")
        # create val dataset
        _ = datastruc.LocDataset(
            os.path.join(input_folder_val, input_folder),
            val_folder,
            config["label_level"],
            val_pre_filter,
            config["save_on_gpu"],
            None,  # transform
            None,  # pre-transform
            config["feat"],
            min_feat,
            max_feat,
            config["fov_x"],
            config["fov_y"],
            kneighbours=config["kneighbours"],
        )

        print("Test set...")
        # create test dataset
        _ = datastruc.LocDataset(
            os.path.join(input_folder_test, input_folder),
            test_folder,
            config["label_level"],
            test_pre_filter,
            config["save_on_gpu"],
            None,  # transform
            None,  # pre-transform
            config["feat"],
            min_feat,
            max_feat,
            config["fov_x"],
            config["fov_y"],
            kneighbours=config["kneighbours"],
        )

        # save yaml file
        yaml_save_loc = os.path.join(project_directory, "process.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
