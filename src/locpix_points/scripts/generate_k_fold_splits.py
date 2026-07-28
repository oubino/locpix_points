"""Generate splits for k-fold

Recipe :
    1. Initialise folds
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import yaml
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

# from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit


def extract_groups(file_list: list[str], prefix: str):
    """Extract groups from names in an input file list. Finds integers
    representing groups after a prefix.

    Args:
        file_list:
            List of files.
        prefix:
            String occuring before group identifier in the filename.
            Group identifier is an integer.

    Returns:
        group_ids:
            List of the groups to which the files belong. Elements are
            str(int).
    """
    group_ids = []
    for file in file_list:
        match = re.search(r"PAT(\d+)", file)  # {re.escape(prefix)}
        group_id = match.group(1) if match else None
        group_ids.append(group_id)
    return(group_ids)


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

    parser.add_argument(
        "-g",
        "--groups",
        action="store",
        type=str,
        default=None,
        help="if present, use non-overlapping groups in the test sets"
        " so that the same group is not present in the test and"
        " training/validation sets. The string given should be a prefix in the"
        " filenames before integers that represent the group."
        " e.g. type \"-g PAT\" for filenames like PAT1, PAT2, etc."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="if present some extra information is shown about the splits"
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
        print(
            "k_fold.yaml already exists, to overwrite provide the --force flag"
        )
        sys.exit(1)
    else:
        # proceed to splitting the data
        splits = {}
        n_splits = args.split

        # Get input file files and ground truth labels
        file_list = os.listdir(os.path.join(project_directory, "preprocessed/gt_label"))
        targets = []

        for file in file_list:
            target = pq.read_table(
                os.path.join(project_directory, "preprocessed/gt_label", file)
            )
            gt_label = int(target.schema.metadata[b"gt_label"])
            targets.append(gt_label)

        # Prepare k-fold split, include groups if present
        # Random splits
        if args.groups is None:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            kfsplit = kf.split(file_list, targets)
        else:
            kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
            print(f"Extracting: {args.groups}")
            group_ids = extract_groups(file_list, args.groups)
            kfsplit = kf.split(file_list, targets, group_ids)

        # Generate folds
        train_folds = []
        val_folds = []
        test_folds = []

        for i, (train_index, test_indices) in enumerate(kfsplit):
            print(f"Fold {i}")
            # Option to see test fold groups
            if args.verbose and args.groups is not None:
                test_groups = np.unique(np.array(group_ids)[test_indices])
                print(f"  Test labels: {test_groups}")

            # Ensure the same file is not going into training and testing sets
            if any(i in train_index for i in test_indices):
                raise ValueError("Should not share common values!!!")

            # Split training set into train/val: 80/20
            train_indices, val_indices = train_test_split(
                train_index,
                test_size=0.2,
                shuffle=True,
                stratify=[targets[idx] for idx in train_index],
            )

            # Optional info
            if args.verbose:
                print(
                    f"# training files: {len(train_indices)}\n"
                    f"# validation files: {len(val_indices)}\n"
                    f"# test files: {len(test_indices)}\n"
                )

            train_folds.append([file_list[idx] for idx in train_indices])
            val_folds.append([file_list[idx] for idx in val_indices])
            test_folds.append([file_list[idx] for idx in test_indices])

        for index, train_fold in enumerate(train_folds):
            val_fold = val_folds[index]
            test_fold = test_folds[index]

            # Ensure the same file is not in one than on of
            # training/validation/testing sets
            if any(i in train_fold for i in val_fold):
                raise ValueError("Should not share common values train/val")
            if any(i in val_fold for i in train_fold):
                raise ValueError("Should not share common values val/train")

            if any(i in train_fold for i in test_fold):
                raise ValueError("Should not share common values train/test")
            if any(i in test_fold for i in train_fold):
                raise ValueError("Should not share common values test/train")

            if any(i in test_fold for i in val_fold):
                raise ValueError("Should not share common values test/val")
            if any(i in val_fold for i in test_fold):
                raise ValueError("Should not share common values val/test")

            if args.groups is not None:
            # Ensure groups in test set are not present in
            # training or validation sets
                train_patients = []
                val_patients = []
                test_patients = []
                extract_groups(train_fold, train_patients)
                extract_groups(val_fold, val_patients)
                extract_groups(test_fold, test_patients)

                if any(i in train_patients for i in test_patients):
                    raise ValueError("Should not share common values train/test")
                if any(i in test_patients for i in train_patients):
                    raise ValueError("Should not share common values test/train")
                if any(i in test_patients for i in val_patients):
                    raise ValueError("Should not share common values test/val")
                if any(i in val_patients for i in test_patients):
                    raise ValueError("Should not share common values val/test")

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
