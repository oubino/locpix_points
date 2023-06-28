"""Process recipe

Recipe :
    1. Create dataset
    2. Process dataset - pre-transform and save to .pt
"""
import random
import os
import yaml
from heptapods.data_loading import datastruc
from functools import partial
import argparse
import json
import time
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

def main():

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Preprocess the data for\
        further processing."
    )

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
                             for processing",
        required=True,
    )

    args = parser.parse_args()

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
    processed_dir_root = os.path.join(project_directory, "output/processed")

    # split into train/val/test using pre filter
    file_list = os.listdir(os.path.join(project_directory, "preprocessed/annotated"))
    file_list = [file.removesuffix('.parquet') for file in file_list]
    random.shuffle(file_list)
    # split into train/test/val
    train_length = int(len(file_list) * config['train_ratio'])
    test_length = int(len(file_list) * config['test_ratio'])
    val_length = len(file_list) - train_length - test_length
    train_list = file_list[0:train_length]
    val_list = file_list[train_length:train_length + val_length]
    test_list = file_list[train_length + val_length: len(file_list)]

    # folders

    train_folder = os.path.join(processed_dir_root, 'train')
    val_folder = os.path.join(processed_dir_root, 'val')
    test_folder = os.path.join(processed_dir_root, 'test')
    # if output directory not present create it
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # bind arguments to functions
    train_pre_filter = partial(pre_filter, inclusion_list=train_list)
    val_pre_filter = partial(pre_filter, inclusion_list=val_list)
    test_pre_filter = partial(pre_filter, inclusion_list=test_list)

    # TODO: #3 Add in pre-transforms to process @oubino

    # create train dataset
    trainset = datastruc.SMLMDataset(config['hetero'],
                                     os.path.join(project_directory, "preprocessed/annotated"),
                                     train_folder,
                                     transform=None,
                                     pre_transform=None,
                                     # e.g. pre_transform =
                                     # T.RadiusGraph(r=0.0000003,
                                     # max_num_neighbors=1),
                                     pre_filter=train_pre_filter)

    # create val dataset
    valset = datastruc.SMLMDataset(config['hetero'],
                                   os.path.join(project_directory, "preprocessed/annotated"),
                                   val_folder,
                                   transform=None,
                                   pre_transform=None,
                                   pre_filter=val_pre_filter)

    # create test dataset
    testset = datastruc.SMLMDataset(config['hetero'],
                                    os.path.join(project_directory, "preprocessed/annotated"),
                                    test_folder,
                                    transform=None,
                                    pre_transform=None,
                                    pre_filter=test_pre_filter)
    

if __name__ == "__main__":
    main
