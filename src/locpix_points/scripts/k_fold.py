"""K-fold recipe

Recipe :
    1. Initialise folds
    2. Process
    3. Train
"""

import argparse
import os
import pyarrow.parquet as pq
import json
import time
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
import wandb

from locpix_points.scripts.evaluate import main as main_eval
from locpix_points.scripts.process import main as main_process
from locpix_points.scripts.train import main as main_train


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If specify random split but also have a config file"""

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

    parser.add_argument(
        "-s",
        "--split",
        action="store",
        type=int,
        help="if present then split the data into number of folds specified",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="if present then forces the split and overwrites",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # initiailse config
    config = None

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

    # split data
    if config is None:
        splits_in_config = False
    else:
        try:
            config["splits"]
            splits_in_config = True
        except KeyError:
            splits_in_config = False

    if args.split is None:
        if not splits_in_config:
            raise ValueError("Should be splits in the config")
        else:
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
    else:
        if splits_in_config and not args.force:
            raise ValueError("If want to overwrite specify force argument")
        else:
            splits = {}
            # split randomly
            n_splits = args.split
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            file_list = os.listdir(
                os.path.join(project_directory, "preprocessed/gt_label")
            )
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
            if config is not None:
                raise ValueError("Config should be none")
            config = {}
            config["splits"] = splits

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

        # print('Cleaning up')

        # clean up process folder check it first during debugging
        # keep_files = ['file_map.csv', 'pre_filter.pt', 'pre_transform.pt']
        # keep_files = []
        # train_files = os.listdir(f'{args.project_directory}/processed/fold_{index}/train')
        # val_files = os.listdir(f'{args.project_directory}/processed/fold_{index}/val')
        # test_files = os.listdir(f'{args.project_directory}/processed/fold_{index}/test')
        # get list of folders to delete
        # train_files = [os.path.join(f'{args.project_directory}/processed/fold_{index}/train', i) for i in train_files if i not in keep_files]
        # val_files = [os.path.join(f'{args.project_directory}/processed/fold_{index}/val', i) for i in val_files if i not in keep_files]
        # test_files = [os.path.join(f'{args.project_directory}/processed/fold_{index}/test', i) for i in test_files if i not in keep_files]

        # for file in train_files:
        #    os.remove(file)
        # for file in val_files:
        #    os.remove(file)
        # for file in test_files:
        #    os.remove(file)

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
