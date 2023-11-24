"""K-fold recipe

Recipe :
    1. Initialise folds
    2. Process
    3. Train
"""

import os
import yaml
import argparse
import time
from sklearn.model_selection import KFold 
from locpix_points.scripts.process import main as main_process
from locpix_points.scripts.train import main as main_train
import shutil

def main(argv=None):
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
        "-r",
        "--random",
        action="store",
        type=int,
        help="whether split should be random",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # initiailse config
    config = None

    # load yaml
    k_fold_yaml = os.path.join(args.config, 'k_fold.yaml')
    with open(k_fold_yaml, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # define data split
    if args.random is not None:
        splits = {}
        # split randomly
        n_splits = args.random
        kf = KFold(n_splits=n_splits, shuffle=True)
        file_list = os.listdir(os.path.join(project_directory, 'preprocessed/gt_label'))
        train_folds = []
        val_folds = []
        test_folds = []
        for (train_index, test_index) in kf.split(file_list):
            train_fold = []
            val_fold = []
            test_fold = []
            # split train into train/val: 80/20
            val_index = int(0.8*len(train_index))
            for index in train_index:
                if index < val_index:
                    train_fold.append(file_list[index])
                else:
                    val_fold.append(file_list[index])
            for index in test_index:
                test_fold.append(file_list[index])
            train_folds.append(train_fold)
            val_folds.append(val_fold)
            test_folds.append(test_fold)
        
        for index, train_fold in enumerate(train_folds):
            val_fold = val_folds[index]
            test_fold = test_folds[index]
            assert train_fold != val_fold
            assert train_fold != test_fold
            assert val_fold != test_fold
            
        # save to config
        splits['train'] = train_folds
        splits['val'] = val_folds
        splits['test'] = test_folds
        if config is not None:
            raise ValueError("Config should be none")
        config = {}
        config['splits'] = splits

    else:
        splits = config['splits'] 
        train_folds = splits['train']
        val_folds = splits['val']
        test_folds = splits['test']
        for index, train_fold in enumerate(train_folds):
            val_fold = val_folds[index]
            test_fold = test_folds[index]
            assert train_fold != val_fold
            assert train_fold != test_fold
            assert val_fold != test_fold

    # for split in splits
    for index, train_fold in enumerate(train_folds):
        print(f'Fold {index}')

        val_fold = val_folds[index]
        test_fold = test_folds[index]

        # process
        main_process(
        [
            "-i",
            args.project_directory,
            "-c",
            f"{args.config}/process.yaml",
            "-o",
            f"processed/fold_{index}"
        ]
        )

        # train
        main_train(
        [
            "-i",
            args.project_directory,
            "-c",
            f"{args.config}/train.yaml",
            "-p",
            f"processed/fold_{index}",
            "-m",
            f"models/fold_{index}"
        ]
        )

        print('Cleaning up')

        # clean up process folder check it first during debugging
        #keep_files = ['file_map.csv', 'pre_filter.pt', 'pre_transform.pt']
        keep_files = []
        train_files = os.listdir(f'{args.project_directory}/processed/fold_{index}/train')
        val_files = os.listdir(f'{args.project_directory}/processed/fold_{index}/val')
        test_files = os.listdir(f'{args.project_directory}/processed/fold_{index}/test')
        # get list of folders to delete
        train_files = [os.path.join(f'{args.project_directory}/processed/fold_{index}/train', i) for i in train_files if i not in keep_files]
        val_files = [os.path.join(f'{args.project_directory}/processed/fold_{index}/val', i) for i in val_files if i not in keep_files]
        test_files = [os.path.join(f'{args.project_directory}/processed/fold_{index}/test', i) for i in test_files if i not in keep_files]

        for file in train_files:
            os.remove(file)
        for file in val_files:
            os.remove(file)
        for file in test_files:
            os.remove(file)
        
        # remove directories
        os.rmdir(f'{args.project_directory}/processed/fold_{index}/train')
        os.rmdir(f'{args.project_directory}/processed/fold_{index}/val')
        os.rmdir(f'{args.project_directory}/processed/fold_{index}/test')
        os.rmdir(f'{args.project_directory}/processed/fold_{index}')

    # save config file to folder and wandb
    yaml_save_loc = os.path.join(project_directory, f"k_fold.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
