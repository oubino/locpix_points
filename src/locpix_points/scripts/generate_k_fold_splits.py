"""Generate splits for k-fold

Recipe :
    1. Initialise folds
"""

import argparse
import json
import os
import pyarrow.parquet as pq
import time
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If have a config file and don't provide force argument"""

    # parse arugments
    parser = argparse.ArgumentParser(description="Generate k-fold splits")

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
        help="the location of the configuaration folder\
                which has process.yaml, train.yaml and k_fold.yaml",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--split",
        action="store",
        type=int,
        help="if present then split the data into number of folds specified",
        required=True,
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="if present then forces the split and overwrites",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # metadata
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

    # check for presence of k_fold.yaml
    k_fold_yaml = os.path.join(args.config, "k_fold.yaml")
    if os.path.exists(k_fold_yaml) and not args.force:
        raise ValueError(
            "k_fold.yaml already exists, to overwrite provide the --force flag"
        )
    else:
        # proceed to splitting the data
        splits = {}
        # split randomly
        n_splits = args.split
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        file_list = os.listdir(os.path.join(project_directory, "preprocessed/gt_label"))
        targets = []
        for file in file_list:
            target = pq.read_table(
                os.path.join(project_directory, "preprocessed/gt_label", file)
            )
            gt_label = int(target.schema.metadata[b"gt_label"])
            targets.append(gt_label)
        train_folds = []
        val_folds = []
        test_folds = []
        for train_index, test_indices in kf.split(file_list, targets):
            if any(i in train_index for i in test_indices):
                raise ValueError("Should not share common values!!!")

            # split train into train/val: 80/20
            train_indices, val_indices = train_test_split(
                train_index,
                test_size=0.2,
                shuffle=True,
                stratify=[targets[idx] for idx in train_index],
            )

            train_folds.append([file_list[idx] for idx in train_indices])
            val_folds.append([file_list[idx] for idx in val_indices])
            test_folds.append([file_list[idx] for idx in test_indices])

        for index, train_fold in enumerate(train_folds):
            val_fold = val_folds[index]
            test_fold = test_folds[index]

            if any(i in train_fold for i in val_fold):
                raise ValueError("Should not share common values 1")
            if any(i in val_fold for i in train_fold):
                raise ValueError("Should not share common values 2")

            if any(i in train_fold for i in test_fold):
                raise ValueError("Should not share common values 3")
            if any(i in test_fold for i in train_fold):
                raise ValueError("Should not share common values 4")

            if any(i in test_fold for i in val_fold):
                raise ValueError("Should not share common values 5")
            if any(i in val_fold for i in test_fold):
                raise ValueError("Should not share common values 6")

        # save to config
        splits["train"] = train_folds
        splits["val"] = val_folds
        splits["test"] = test_folds
        config = {}
        config["splits"] = splits

    # save config file to project directory
    yaml_save_loc = os.path.join(project_directory, args.config, "k_fold.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
