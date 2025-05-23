"""Evaluate recipe

Recipe :
    1. Load in model
    2. Initialise dataloader
    3. Evaluate on test set
"""

import argparse
import json
import os
import time

import pandas as pd
import torch
import torch_geometric.loader as L
import yaml
from torchsummary import summary

import wandb
from locpix_points.data_loading import datastruc
from locpix_points.evaluation import evaluate
from locpix_points.models import model_choice

# import torch
# import torch_geometric.transforms as T


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If GPU/CPU argument incorrectly specified"""

    # parse arugments
    parser = argparse.ArgumentParser(description="Evaluating")

    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="location of the project directory",
        required=True,
    )

    parser.add_argument(
        "-p",
        "--processed_directory",
        action="store",
        type=str,
        help="the location of the processed files\
                if not specified then defaults to\
                project_directory/processed",
    )

    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for evaluating",
        required=True,
    )

    parser.add_argument(
        "-m",
        "--model_loc",
        action="store",
        type=str,
        help="path to model for evaluation",
        required=True,
    )

    parser.add_argument(
        "-w",
        "--wandbstarted",
        action="store_true",
        help="if specified then wandb has already been initialised",
    )

    parser.add_argument(
        "-n",
        "--run_name",
        action="store",
        type=str,
        help="name of the run in wandb",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # load yaml
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in config
    load_data_from_gpu = config["load_data_from_gpu"]
    eval_on_gpu = config["eval_on_gpu"]
    num_classes = config["num_classes"]

    # load metadata
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        project_name = metadata["project_name"]
        dataset_name = metadata["dataset_name"]
        user = metadata["user"]

    # if data is on gpu then don't need to pin memory
    # and this causes errors if try
    if load_data_from_gpu is True:
        pin_memory = False
    elif load_data_from_gpu is False:
        pin_memory = True
    else:
        raise ValueError("load_data_from_gpu should be True or False")

    # define device
    if eval_on_gpu is True and not torch.cuda.is_available():
        raise ValueError("No gpu available, you should run on cpu instead")
    elif eval_on_gpu is True and torch.cuda.is_available():
        device = torch.device("cuda")
    elif eval_on_gpu is False:
        device = torch.device("cpu")
    else:
        raise ValueError("Specify cpu or gpu !")

    # folder
    if args.processed_directory is not None:
        processed_directory = os.path.join(project_directory, args.processed_directory)
    else:
        processed_directory = os.path.join(project_directory, "processed")

    test_folder = os.path.join(processed_directory, "test")

    if config["model"] in [
        "locclusternet",
        "clusternet",
        "clustermlp",
        "locnetonly_pointnet",
        "locnetonly_pointtransformer",
    ]:
        test_set = datastruc.ClusterLocDataset(
            None,  # raw_loc_dir_root
            None,  # raw_cluster_dir_root
            test_folder,  # processed_dir_root
            label_level=config["label_level"],  # label_level
            pre_filter=None,  # pre_filter
            save_on_gpu=load_data_from_gpu,  # gpu
            transform=None,  # transform
            pre_transform=None,  # pre_transform
            loc_feat=None,
            cluster_feat=None,
            min_feat_locs=None,
            max_feat_locs=None,
            min_feat_clusters=None,
            max_feat_clusters=None,
            kneighboursclusters=None,
            fov_x=None,
            fov_y=None,
            kneighbourslocs=None,
            range_xy=False,
        )

    elif config["model"] in ["loconlynet"]:
        # load in test dataset
        test_set = datastruc.LocDataset(
            None,  # raw_loc_dir_root
            test_folder,  # processed_dir_root
            label_level=config["label_level"],  # label_level
            pre_filter=None,  # pre_filter
            save_on_gpu=load_data_from_gpu,  # gpu
            transform=None,  # transform
            pre_transform=None,  # pre_transform
            feat=None,
            min_feat=None,
            max_feat=None,
            fov_x=None,
            fov_y=None,
            kneighbours=None,
            range_xy=False,
        )

    else:
        raise ValueError("Model not defined")

    test_loader = L.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
    )
    for _, data in enumerate(test_loader):
        first_item = data
    if config["model"] in [
        "locclusternet",
        "clusternet",
        "clustermlp",
        "locnetonly_pointnet",
        "locnetonly_pointtransformer",
    ]:
        dim = first_item["locs"].pos.shape[-1]
    elif config["model"] in ["loconlynet"]:
        dim = first_item.pos.shape[-1]
    else:
        raise ValueError("Model not defined")
    # initialise model
    model = model_choice(
        config["model"],
        # this should parameterise the chosen model
        config[config["model"]],
        dim=dim,
        device=device,
    )

    print("\n")
    print("Loading in best model")
    print("\n")
    model.load_state_dict(torch.load(args.model_loc, weights_only=False))
    model.to(device)

    # model summary
    print("\n")
    print("---- Model summary ----")
    print("\n")
    number_nodes = 1000  # this is just for summary, has no bearing on training
    summary(
        model,
        input_size=(test_set.num_node_features, number_nodes),
        batch_size=1,
    )

    if not args.wandbstarted:
        wandb.login()

    # initialise wandb
    if not args.wandbstarted:
        if args.run_name is None:
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project=dataset_name,
                # set the entity to the user
                entity=user,
                # group by dataset
                group=project_name,
                # track hyperparameters and run metadata
                config={
                    "model": args.model_loc,
                    "architecture": model.name,
                    "dataset": dataset_name,
                },
            )
        else:
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project=dataset_name,
                # set the entity to the user
                entity=user,
                # group by dataset
                group=project_name,
                # track hyperparameters and run metadata
                config={
                    "model": args.model_loc,
                    "architecture": model.name,
                    "dataset": dataset_name,
                },
                name=args.run_name,
            )

    else:
        wandb.config["model"] = args.model_loc

    print("\n")
    print("---- Predict on test set... ----")
    print("\n")
    metrics, roc_metrics = evaluate.make_prediction_test(
        model,
        test_loader,
        device,
        num_classes,
        explain=False,
    )

    for key, value in metrics.items():
        print(f"{key} : {value.item()}")

    # log metrics
    wandb.log(metrics)

    for i in range(num_classes):
        FPR = roc_metrics[f"TestFPR_{i}"].cpu()
        TPR = roc_metrics[f"TestTPR_{i}"].cpu()
        THRESH = roc_metrics[f"TestThreshold_{i}"].cpu()
        df = pd.DataFrame({"FPR": FPR, "TPR": TPR, "THRESH": THRESH})
        roc_table = wandb.Table(dataframe=df)
        wandb.log({f"Test_ROC_{i}": roc_table})

    time_o = time.gmtime(time.time())
    time_o = (
        f"hhmm_{time_o[3]}_{time_o[4]}_ddmmyyyy_{time_o[2]}_{time_o[1]}_{time_o[0]}"
    )

    # save config file to folder and wandb
    # yaml_save_loc = os.path.join(project_directory, f"evaluate_{time_o}.yaml")
    # with open(yaml_save_loc, "w") as outfile:
    #    yaml.dump(config, outfile)
    yaml_save_loc = os.path.join(wandb.run.dir, f"evaluate_{time_o}.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)

    # exit wandb
    if not args.wandbstarted:
        wandb.finish()


if __name__ == "__main__":
    main()
