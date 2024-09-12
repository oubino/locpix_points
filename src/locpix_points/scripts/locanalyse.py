"""Feature analysis module

Module takes in the processed localisations.

Then rather than passing through the model as usual - calculating the edge index dynamically

We construct the edge index statically for the initial localisations.

We mandate that all ratios set to 1.

We can then perform subgraphX and attention (if relevant) analysis.

We could also implement other point cloud based explainability methods.

Config file at top specifies the analyses we want to run"""

import argparse
import json
import os
import time
import yaml

from locpix_points.data_loading import datastruc
from locpix_points.models import model_choice

import torch
import torch_geometric.loader as L


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arugments
    parser = argparse.ArgumentParser(description="Analyse features")

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
                             for evaluating",
        required=True,
    )

    parser.add_argument(
        "-a",
        "--automatic",
        action="store_true",
        help="if present then there should be only one model present in the folder"
        "which we load in",
    )

    parser.add_argument(
        "-f",
        "--final_test",
        action="store_true",
        help="if specified then running final test",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    label_map = config["label_map"]

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

    # make output folder
    output_folder = os.path.join(project_directory, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- Analyse loc features -------
    analyse_locs(project_directory, config, args)


def analyse_locs(project_directory, config, args):
    """Analyse the localisations

    Args:
        project_directory (str): Location of the project directory
        config (dict): Configuration for this script
        args (dict): Arguments passed to this script

    Raises:
        ValueError: If device specified is neither cpu or gpu OR
            if attention to examine is not correctly specified OR
            if encoder not loc or cluster or fov
    """

    # ----------------------------

    if config["device"] == "gpu":
        device = torch.device("cuda")
    elif config["device"] == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("Device should be cpu or gpu")

    # load in gt_label_map
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        gt_label_map = metadata["gt_label_map"]

    gt_label_map = {int(key): val for key, val in gt_label_map.items()}

    model_type = config["model"]

    # only works for locclusternet at the moment
    assert model_type == "loconlynet"

    # initialise model
    model = model_choice(
        model_type,
        # this should parameterise the chosen model
        config[model_type],
        device=device,
    )

    # load in best model
    print("\n")
    print("Loading in best model")
    print("\n")
    if not args.final_test:
        # needs to be from same fold as below
        fold = config["fold"]
    model_name = config["model_name"]
    if not args.automatic:
        if not args.final_test:
            model_loc = os.path.join(
                project_directory, "models", f"fold_{fold}", model_name
            )
        else:
            model_loc = os.path.join(project_directory, "models", model_name)
    elif args.automatic:
        if not args.final_test:
            model_dir = os.path.join(project_directory, "models", f"fold_{fold}")
        else:
            model_dir = os.path.join(project_directory, "models")
        model_list = os.listdir(model_dir)
        assert len(model_list) == 1
        model_name = model_list[0]
        model_loc = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(model_loc))
    model.to(device)
    model.eval()

    if not args.final_test:
        train_folder = os.path.join(
            project_directory, "processed", f"fold_{fold}", "train"
        )
        val_folder = os.path.join(project_directory, "processed", f"fold_{fold}", "val")
        test_folder = os.path.join(
            project_directory, "processed", f"fold_{fold}", "test"
        )
    else:
        train_folder = os.path.join(project_directory, "processed", "train")
        val_folder = os.path.join(project_directory, "processed", "val")
        test_folder = os.path.join(project_directory, "processed", "test")

    # initialise train/validation and test sets
    train_set = datastruc.LocDataset(
        None,  # raw_loc_dir_root
        train_folder,  # processed_dir_root
        label_level=None,
        pre_filter=None,  # pre_filter
        save_on_gpu=False,  # gpu
        transform=None,  # transform
        pre_transform=None,  # pre_transform
        feat=None,
        min_feat=None,
        max_feat=None,
        fov_x=None,
        fov_y=None,
        kneighbours=None,
    )

    # load in val dataset
    val_set = datastruc.LocDataset(
        None,  # raw_loc_dir_root
        val_folder,  # processed_dir_root
        label_level=None,
        pre_filter=None,  # pre_filter
        save_on_gpu=False,  # gpu
        transform=None,
        pre_transform=None,
        feat=None,
        min_feat=None,
        max_feat=None,
        fov_x=None,
        fov_y=None,
        kneighbours=None,
    )

    # load in test dataset
    test_set = datastruc.LocDataset(
        None,  # raw_loc_dir_root
        test_folder,  # processed_dir_root
        label_level=None,
        pre_filter=None,  # pre_filter
        save_on_gpu=False,  # gpu
        transform=None,  # transform
        pre_transform=None,  # pre_transform
        feat=None,
        min_feat=None,
        max_feat=None,
        fov_x=None,
        fov_y=None,
        kneighbours=None,
    )

    # aggregate cluster features into collated df
    if not args.final_test:
        train_set = torch.utils.data.ConcatDataset([train_set, val_set, test_set])
        test_set = None
    else:
        train_set = torch.utils.data.ConcatDataset([train_set, val_set])
        test_set = test_set

    # dataloader
    train_loader = L.DataLoader(
        train_set,
        batch_size=128,  # change in config
        shuffle=True,
        drop_last=True,
    )

    for _, data in enumerate(train_loader):
        pointnet = False
        pointtransformer = False

        if pointnet:
            pos = data.pos
            batch = data.batch
            r = ...
            k = ...

            row, col = radius(
                pos,
                pos,
                self.r,
                batch,
                batch,
                max_num_neighbors=self.k + 1,
            )
            edge_index = torch.stack([col, row], dim=0)
            assert contains_self_loops(edge_index)

            data.edge_index = edge_index

        elif pointtransformer:
            print("bob")

        # run through model with static=True and edge index defined

        # run through model with static=False and edge index not defined

        # check are teh same
        # break

    # Caluclate edge index

    # Pass through the static version of PointNet or PointTransformer

    # Mandate that all ratios = 1

    # Attention averaged over each edge

    # SubgraphX

    # We could also implement other point cloud based explainability methods.

    print(train_set)
    print(val_set)
    print(test_set)


if __name__ == "__main__":
    main()
