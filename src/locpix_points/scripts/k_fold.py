"""K-fold recipe

Recipe :
    1. Loads folds
    2. Process
    3. Train
"""

import argparse
import os
import json
import time
import yaml
import wandb

from locpix_points.scripts.evaluate import main as main_eval
from locpix_points.scripts.process import main as main_process
from locpix_points.scripts.train import main as main_train


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arugments
    parser = argparse.ArgumentParser(description="Training")

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

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # load yaml
    k_fold_yaml = os.path.join(args.config, "k_fold.yaml")
    with open(k_fold_yaml, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

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
        # Get config params
        project_name = metadata["project_name"]
        dataset_name = metadata["dataset_name"]
        user = metadata["user"]
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # login to wandb
    wandb.login()

    # Load splits
    splits = config["splits"]
    train_folds = splits["train"]
    val_folds = splits["val"]
    test_folds = splits["test"]
    for index, train_fold in enumerate(train_folds):
        val_fold = val_folds[index]
        test_fold = test_folds[index]
        assert train_fold != val_fold
        assert train_fold != test_fold
        assert val_fold != test_fold

    # for split in splits
    for index, train_fold in enumerate(train_folds):
        print(f"Fold {index}")

        val_fold = val_folds[index]
        test_fold = test_folds[index]

        train_fold = [x.rstrip(".parquet") for x in train_fold]
        val_fold = [x.rstrip(".parquet") for x in val_fold]
        test_fold = [x.rstrip(".parquet") for x in test_fold]

        # initialise wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project=dataset_name,
            # set the entity to the user
            entity=user,
            # group by dataset
            group=project_name,
            # name for this run
            name=f"fold_{index}",
        )

        # process
        main_process(
            [
                "-i",
                args.project_directory,
                "-c",
                f"{args.config}/process.yaml",
                "-o",
                f"processed/fold_{index}",
                "-k",
                train_fold,
                "-k",
                val_fold,
                "-k",
                test_fold,
            ]
        )

        # train
        model_path = main_train(
            [
                "-i",
                args.project_directory,
                "-c",
                f"{args.config}/train.yaml",
                "-p",
                f"processed/fold_{index}",
                "-m",
                f"models/fold_{index}",
                "-w",
            ]
        )

        # evaluate
        main_eval(
            [
                "-i",
                args.project_directory,
                "-p",
                f"processed/fold_{index}",
                "-c",
                f"{args.config}/evaluate.yaml",
                "-m",
                model_path,
                "-w",
            ]
        )

        wandb.finish()

        print("Cleaning up")

        # clean up process folder check it first during debugging
        keep_files = ["file_map.csv", "pre_filter.pt", "pre_transform.pt"]
        # keep_files = []
        train_files = os.listdir(
            f"{args.project_directory}/processed/fold_{index}/train"
        )
        val_files = os.listdir(f"{args.project_directory}/processed/fold_{index}/val")
        test_files = os.listdir(f"{args.project_directory}/processed/fold_{index}/test")
        # get list of folders to delete
        train_files = [
            os.path.join(f"{args.project_directory}/processed/fold_{index}/train", i)
            for i in train_files
            if i not in keep_files
        ]
        val_files = [
            os.path.join(f"{args.project_directory}/processed/fold_{index}/val", i)
            for i in val_files
            if i not in keep_files
        ]
        test_files = [
            os.path.join(f"{args.project_directory}/processed/fold_{index}/test", i)
            for i in test_files
            if i not in keep_files
        ]

        for file in train_files:
            os.remove(file)
        for file in val_files:
            os.remove(file)
        for file in test_files:
            os.remove(file)

        # remove directories
        # os.rmdir(f'{args.project_directory}/processed/fold_{index}/train')
        # os.rmdir(f'{args.project_directory}/processed/fold_{index}/val')
        # os.rmdir(f'{args.project_directory}/processed/fold_{index}/test')
        # os.rmdir(f'{args.project_directory}/processed/fold_{index}')

    # save config file to folder and wandb
    yaml_save_loc = os.path.join(project_directory, "k_fold.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
