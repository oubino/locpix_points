"""Process recipe

Recipe :
    1. Create dataset
    2. Process dataset - pre-transform and save to .pt
"""
import random
import os
import yaml
from locpix_points.data_loading import datastruc
from functools import partial
import argparse
import json
import time
import warnings
import torch
import ast
import polars as pl
import numpy as np

# import torch_geometric.transforms as T


def pre_filter(data, inclusion_list=[]):
    """Takes in data item and returns whether
    it should be included in final dataset
    i.e. 1 - yes ; 0 - no

    Args:
        data (torch.geometric.data) : The pytorch
            geometric dataitem part of the dataset
        inclusion_list (list) : List of names
            indicating which data should be included"""

    if data.name in inclusion_list:
        return 1
    else:
        return 0
    
def load_pre_filter(path):
    """Load in a pre-filter from previous run
    
    Args:
        path (string) : Path to the pre-filter.pt"""

    pre_filter = torch.load(path)
    if pre_filter.startswith("functools.partial(<function>, inclusion_list=") and pre_filter[-1] == ")":
        pre_filter = pre_filter.removeprefix("functools.partial(<function>, inclusion_list=")
        pre_filter = pre_filter[:-1:]
        return ast.literal_eval(pre_filter)  
    else:
        raise ValueError("Unknown pre-filter type")

def minmax(config, feat_str, file_directory, train_list):
    raise ValueError("Check correct")
    if type(config[feat_str]) is list:
        for index, file in enumerate(train_list):
            df = pl.read_parquet(os.path.join(file_directory, file + '.parquet'))
            min_df = df.select(pl.col(config[feat_str]).min())
            max_df = df.select(pl.col(config[feat_str]).max())
            if index == 0:
                min_vals = min_df.to_numpy()[0]
                max_vals = max_df.to_numpy()[0]
            else:
                min_vals = np.min((min_vals, min_df.to_numpy()[0]), axis=0)
                max_vals = np.max((max_vals, max_df.to_numpy()[0]), axis=0)
    else:
        raise NotImplementedError

    min_vals = dict(zip(config[feat_str], min_vals))
    max_vals = dict(zip(config[feat_str], max_vals))

    return min_vals, max_vals

def main():

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
        nargs='+',
        action='append',
        default=None,
        help="list of lists, list[0]=train files, list[1] = val files, list[2] = test files",
    )

    args = parser.parse_args()

    print(args.manual_split)

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
    processed_dir_root = os.path.join(project_directory, "processed")

    # split into train/val/test using pre filter

    # split randomly
    if args.split is None and args.manual_split is None:
        warnings.warn("Is this the correct file list")
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

    elif args.split is not None and args.manual_split is None:
        warnings.warn("Known omission is if pre-transform is done to dataset"
                      "this is not currently also done to this dataset as well")
        
        train_list_path = os.path.join(args.split, "processed/train/pre_filter.pt")
        val_list_path = os.path.join(args.split, "processed/val/pre_filter.pt")
        test_list_path = os.path.join(args.split, "processed/test/pre_filter.pt")

        train_list = load_pre_filter(train_list_path)
        val_list = load_pre_filter(val_list_path)
        test_list = load_pre_filter(test_list_path)
    
    elif args.split is None and args.manual_split is not None:
        train_list = args.manual_split[0]
        val_list = args.manual_split[1]
        test_list = args.manual_split[2]

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
    
    # calculate min/max features 
    if config['model'] == 'ClusterLoc':

        file_directory = os.path.join(project_directory, 'preprocessed/featextract/locs')
        min_feat_locs, max_feat_locs = minmax(config, 'loc_feat', file_directory, train_list)
        file_directory = os.path.join(project_directory, 'preprocessed/featextract/clusters')
        min_feat_clusters, max_feat_clusters = minmax(config, 'cluster_feat', project_directory, train_list)

        print("Train set...")
        # create train dataset
        trainset = datastruc.ClusterLocDataset(
            os.path.join(project_directory, 'preprocessed/featextract/locs'),
            os.path.join(project_directory, 'preprocessed/featextract/clusters'),
            train_folder,
            config['label_level'],
            train_pre_filter,
            config['gpu'],
            None, # transform
            None, # pre-transform
            config['loc_feat'],
            config['cluster_feat'],
            min_feat_locs,
            max_feat_locs,
            min_feat_clusters,
            max_feat_clusters,
            config['kneighbours'],
        )

        print("Val set...")
        # create val dataset
        valset = datastruc.SMLMDataset(
            os.path.join(project_directory, 'preprocessed/featextract/locs'),
            os.path.join(project_directory, 'preprocessed/featextract/clusters'),
            val_folder,
            config['label_level'],
            val_pre_filter,
            config['gpu'],
            None, # transform
            None, # pre-transform
            config['loc_feat'],
            config['cluster_feat'],
            min_feat_locs,
            max_feat_locs,
            min_feat_clusters,
            max_feat_clusters,
            config['kneighbours'],
        )

        print("Test set...")
        # create test dataset
        testset = datastruc.SMLMDataset(
            os.path.join(project_directory, 'preprocessed/featextract/locs'),
            os.path.join(project_directory, 'preprocessed/featextract/clusters'),
            test_folder,
            config['label_level'],
            test_pre_filter,
            config['gpu'],
            None, # transform
            None, # pre-transform
            config['loc_feat'],
            config['cluster_feat'],
            min_feat_locs,
            max_feat_locs,
            min_feat_clusters,
            max_feat_clusters,
            config['kneighbours'],
        )

        # save yaml file
        yaml_save_loc = os.path.join(project_directory, "process.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)
    
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
