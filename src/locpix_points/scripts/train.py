"""Train recipe

Recipe :
    1. Initialise dataset
    2. Initialise dataloader
    3. Train...
"""

import argparse
import os
import json
import time

import pandas as pd
import torch.optim
import torch_geometric.loader as L
import yaml

# from torchsummary import summary

import wandb
from locpix_points.data_loading import datastruc
from locpix_points.evaluation import evaluate
from locpix_points.models import model_choice
from locpix_points.training import train

# import torch
# import torch_geometric.transforms as T


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Returns:
        model_path: Path to the saved model

    Raises:
        ValueError: If GPU argument incorrectly specified"""

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
        help="the location of the .yaml configuaration file\
                             for processing",
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
        "-m",
        "--model_folder",
        action="store",
        type=str,
        help="where to store the models if not specified\
                defaults to project_directory/models",
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
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    train_on_gpu = config["train_on_gpu"]
    load_data_from_gpu = config["load_data_from_gpu"]
    optimiser = config["optimiser"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    num_workers = config["num_workers"]
    loss_fn = config["loss_fn"]
    label_level = config["label_level"]

    # load metadata
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        project_name = metadata["project_name"]
        dataset_name = metadata["dataset_name"]
        user = metadata["user"]

    # define device
    if train_on_gpu is True and not torch.cuda.is_available():
        raise ValueError(
            "No gpu available, can run on cpu\
                         instead"
        )
    elif train_on_gpu is True and torch.cuda.is_available():
        device = torch.device("cuda")
    elif train_on_gpu is False:
        device = torch.device("cpu")
    else:
        raise ValueError("Specify cpu or gpu !")

    # folder
    if args.processed_directory is not None:
        processed_directory = os.path.join(project_directory, args.processed_directory)
    else:
        processed_directory = os.path.join(project_directory, "processed")
    train_folder = os.path.join(processed_directory, "train")
    val_folder = os.path.join(processed_directory, "val")
    test_folder = os.path.join(processed_directory, "test")

    # load in train dataset
    train_set = datastruc.ClusterLocDataset(
        None,  # raw_loc_dir_root
        None,  # raw_cluster_dir_root
        train_folder,  # processed_dir_root
        label_level=config["label_level"],  # label_level
        pre_filter=None,  # pre_filter
        save_on_gpu=load_data_from_gpu,  # gpu
        transform=config["transforms"],  # transform
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
    )

    print(f"Length of train dataset {len(train_set)}")

    # load in val dataset
    val_set = datastruc.ClusterLocDataset(
        None,  # raw_loc_dir_root
        None,  # raw_cluster_dir_root
        val_folder,  # processed_dir_root
        label_level=config["label_level"],  # label_level
        pre_filter=None,  # pre_filter
        save_on_gpu=load_data_from_gpu,  # gpu
        transform=config["transforms"],  # transform
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
    )

    print(f"Length of validation dataset {len(val_set)}")

    # load in test dataset
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
    )

    print(f"Length of test dataset {len(test_set)}")

    # if data is on gpu then don't need to pin memory
    # and this causes errors if try
    if load_data_from_gpu is True:
        pin_memory = False
    elif load_data_from_gpu is False:
        pin_memory = True
    else:
        raise ValueError("load_data_from_gpu should be True or False")

    # initialise dataloaders
    train_loader = L.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_loader = L.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    # print parameters
    print("\n")
    print("---- Params -----")
    print("\n")
    print("Input features: ", train_set.num_node_features)
    num_classes = max(train_set.num_classes, val_set.num_classes, test_set.num_classes)
    print("Num classes: ", num_classes)
    print("Batch size: ", batch_size)
    print("Epochs: ", epochs)
    num_train_graph = len(train_set)
    print("Number train graphs", num_train_graph)
    num_val_graph = len(val_set)
    print("Number val graphs", num_val_graph)
    for index, data in enumerate(train_loader):
        first_train_item = data
    # nodes = first_train_item.num_nodes
    # label = first_train_item.y
    # if label_level == "node":
    #    assert label.shape[0] == nodes
    # elif label_level == "graph":
    #    print('label', label)
    #    print('label shape', label.shape)
    # line below is incorrect as we have batch dimension as well
    #    assert label.shape == torch.Size([1])
    # else:
    #    raise ValueError("Label level not defined")
    dim = first_train_item["locs"].pos.shape[-1]
    print("Dim", dim)

    # initialise model
    model = model_choice(
        config["model"],
        # this should parameterise the chosen model
        config[config["model"]],
        dim=dim,
        device=device,
    )

    # initialise optimiser
    if optimiser == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # initialise loss function
    if loss_fn == "nll":  # CHANGE
        loss_fn = torch.nn.functional.nll_loss

    if not args.wandbstarted:
        wandb.login()

    # initialise wandb
    if not args.wandbstarted:
        # start a new wandb run to track this script
        if args.run_name is None:
            wandb.init(
                # set the wandb project where this run will be logged
                project=dataset_name,
                # set the entity to the user
                entity=user,
                # group by dataset
                group=project_name,
                # track hyperparameters and run metadata
                config={
                    "learning_rate": lr,
                    "architecture": model.name,
                    "dataset": dataset_name,
                    "epochs": epochs,
                },
            )
        else:
            wandb.init(
                # set the wandb project where this run will be logged
                project=dataset_name,
                # set the entity to the user
                entity=user,
                # group by dataset
                group=project_name,
                # track hyperparameters and run metadata
                config={
                    "learning_rate": lr,
                    "architecture": model.name,
                    "dataset": dataset_name,
                    "epochs": epochs,
                },
                name=args.run_name,
            )
    else:
        wandb.config["learning_rate"] = lr
        wandb.config["architecture"] = model.name
        wandb.config["epochs"] = epochs

    # model summary
    # print("\n")
    # print("---- Model summary (estimate) ----")
    # print("\n")
    # number_nodes = nodes * len(
    #    train_set
    # )  # this is just for summary, has no bearing on training
    # summary(
    #    model,
    #    input_size=(train_set.num_node_features, number_nodes),
    #    batch_size=batch_size,
    # )

    # define save location for model
    if args.model_folder is not None:
        model_folder = os.path.join(project_directory, args.model_folder)
    else:
        model_folder = os.path.join(project_directory, "models")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    time_o = time.gmtime(time.time())
    time_o = f"{time_o[3]}:{time_o[4]}_{time_o[2]}:{time_o[1]}:{time_o[0]}"
    model_path = f"{project_name}_{dataset_name}_{time_o}_.pt"
    model_path = os.path.join(model_folder, model_path)

    # train loop
    print("\n")
    print("---- Training... ----")
    print("\n")
    train.train_loop(  # CHANGE
        epochs,
        model,
        optimiser,
        train_loader,
        val_loader,
        loss_fn,
        device,
        label_level,
        num_train_graph,
        num_val_graph,
        model_path,
    )
    print("\n")
    print("---- Finished training... ----")
    print("Model saved when loss on validation set was lowest")
    print("\n")

    print("\n")
    print("Loading in best model")
    print("\n")
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    print("\n")
    print("---- Predict on train & val set... ----")
    print("\n")
    metrics, roc_metrics = evaluate.make_prediction(
        model,
        optimiser,
        train_loader,
        val_loader,
        device,
        num_classes,
    )

    # log metrics
    wandb.log(metrics)

    # log roc metrics
    for i in range(num_classes):
        for split in ["Train", "Val"]:
            FPR = roc_metrics[f"{split}FPR_{i}"].cpu()
            TPR = roc_metrics[f"{split}TPR_{i}"].cpu()
            THRESH = roc_metrics[f"{split}Threshold_{i}"].cpu()
            df = pd.DataFrame({"FPR": FPR, "TPR": TPR, "THRESH": THRESH})
            roc_table = wandb.Table(dataframe=df)
            wandb.log({f"{split}_ROC_{i}": roc_table})

    # print("\n")
    # print("----- Saving model... ------")
    # if args.processed_directory is not None:
    #    processed_directory = os.path.join(project_directory, args.processed_directory)
    # else:
    #    processed_directory = os.path.join(project_directory, "processed")

    # save config file to folder and wandb
    yaml_save_loc = os.path.join(project_directory, f"train_{time_o}.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)
    yaml_save_loc = os.path.join(wandb.run.dir, f"train_{time_o}.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)

    # exit wandb
    if not args.wandbstarted:
        wandb.finish()

    return model_path


if __name__ == "__main__":
    main()
