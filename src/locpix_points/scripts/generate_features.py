# Load in final version of a model for each fold evaluate on data and measure number of correct for WT

# Imports
import argparse
import os
import polars as pl

import json

import numpy as np
import torch
import torch_geometric.loader as L
import yaml
from torchsummary import summary

from locpix_points.data_loading import datastruc
from locpix_points.models import model_choice

import torch
import warnings


def generate(
    gt_label_map,
    model,
    loader,
    device,
    fold,
    repeats=25,
):
    """Make predictions using the model

    Args:
        gt_label_map (dict) : Map from integers to real label
        model (pytorch geo model) : Model that will make predictiions
        loader (torch dataloader): Dataloader for the
            test dataset
        device (gpu or cpu): Device to evaluate the model
            on
        fold (int) : Which fold is on
        repeats (int) : How many times to sample to generate prediction

    Returns:
        loc_dfs (list) : List of clustres features from LocNet
        cluster_dfs (list) : List of cluster features from ClusterNet
        fov_dfs (list) : List of FOV features agg from ClusterNet
        file_list (list) : Names of files

    Raises:
        NotImplementedError : If model doesn't have attentional aggregation
            final layer"""

    model.to(device)

    # test data
    model.eval()
    file_list = []

    loc_dfs = []
    cluster_dfs = []
    fov_dfs = []

    # a dict to store the activations
    activation = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # register forward hook
    h_0 = model.loc_net.register_forward_hook(getActivation("locencoder"))
    h_1 = model.cluster_net.cluster_encoder_3.register_forward_hook(
        getActivation("clusterencoder")
    )
    try:
        h_2 = model.cluster_net.attention_readout_fn.register_forward_hook(
            getActivation("globalpool")
        )
    except:
        raise NotImplementedError(
            "Only attention as final layer is currently implemented"
        )

    for index, data in enumerate(loader):
        with torch.no_grad():
            # note set to none is meant to have less memory footprint
            # move data to device
            data.to(device)

            gt_label = int(data.y)
            label = gt_label_map[gt_label]

            file_list.append(data.name[0])  # only works cause batch size is 1

            # forward pass - with autocasting
            with torch.autocast(device_type="cuda"):
                loc_data = []
                cluster_data = []
                fov_data = []
                for _ in range(repeats):
                    model(data)
                    loc_data.append(activation["locencoder"].cpu().numpy())
                    cluster_data.append(activation["clusterencoder"].cpu().numpy())
                    fov_data.append(activation["globalpool"].cpu().numpy())

                loc_data = np.mean(np.stack(loc_data), axis=0)
                cluster_data = np.mean(np.stack(cluster_data), axis=0)
                fov_data = np.mean(np.stack(fov_data), axis=0)

                loc_df = pl.DataFrame(loc_data)
                cluster_df = pl.DataFrame(cluster_data)
                fov_df = pl.DataFrame(fov_data)

                loc_df = loc_df.with_columns(pl.lit(label).alias("type"))
                loc_df = loc_df.with_columns(
                    pl.lit(f"{data.name[0]}").alias("file_name")
                )
                loc_df = loc_df.with_columns(pl.lit(f"{fold}").alias("fold"))
                loc_dfs.append(loc_df)

                cluster_df = cluster_df.with_columns(pl.lit(label).alias("type"))
                cluster_df = cluster_df.with_columns(
                    pl.lit(f"{data.name[0]}").alias("file_name")
                )
                cluster_df = cluster_df.with_columns(pl.lit(f"{fold}").alias("fold"))
                cluster_dfs.append(cluster_df)

                fov_df = fov_df.with_columns(pl.lit(label).alias("type"))
                fov_df = fov_df.with_columns(
                    pl.lit(f"{data.name[0]}").alias("file_name")
                )
                fov_df = fov_df.with_columns(pl.lit(f"{fold}").alias("fold"))
                fov_dfs.append(fov_df)

    # remove foward hook
    h_0.remove()
    h_1.remove()
    h_2.remove()

    return loc_dfs, cluster_dfs, fov_dfs, file_list


def main(argv=None):
    # load config

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Generate features from neural network at three levels."
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
        "-r",
        "--repeats",
        type=int,
        help="if provided then perform this number of repeats to generate the prediction",
        required=False,
        default=25,
    )

    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        type=str,
        required=True,
        help="location of the featanalyse config file",
    )

    parser.add_argument(
        "-f",
        "--final_test",
        action="store_true",
        required=False,
        help="if given then is final test",
    )

    args = parser.parse_args(argv)

    if args.final_test:
        raise NotImplementedError("Not implemented yet!")

    config_loc = args.config_file
    project_directory = args.project_directory

    # if data is on gpu then don't need to pin memory
    pin_memory = True

    # define device
    device = torch.device("cuda")

    # load configuration
    with open(config_loc, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in gt_label_map
    gt_label_map = config["label_map"]
    gt_label_map = {int(val): key for key, val in gt_label_map.items()}

    # prepare output folder
    if not os.path.exists(os.path.join(project_directory, f"output")):
        os.makedirs(os.path.join(project_directory, f"output"))

    # load in model name
    model_name = config["model_name"]

    train_loc_dfs = []
    train_cluster_dfs = []
    train_fov_dfs = []

    val_loc_dfs = []
    val_cluster_dfs = []
    val_fov_dfs = []

    test_loc_dfs = []
    test_cluster_dfs = []
    test_fov_dfs = []

    OVERALL_test_file_list = []

    # For each fold
    for fold in range(5):
        # Load in model
        processed_directory = os.path.join(project_directory, f"processed/fold_{fold}")
        model_loc = os.path.join(project_directory, f"models/fold_{fold}", model_name)

        train_folder = os.path.join(processed_directory, "train")
        val_folder = os.path.join(processed_directory, "val")
        test_folder = os.path.join(processed_directory, "test")

        train_files = pl.read_csv(os.path.join(train_folder, "file_map.csv"))[
            "file_name"
        ].to_list()
        val_files = pl.read_csv(os.path.join(val_folder, "file_map.csv"))[
            "file_name"
        ].to_list()
        test_files = pl.read_csv(os.path.join(test_folder, "file_map.csv"))[
            "file_name"
        ].to_list()
        assert len([x for x in test_files if x in train_files]) == 0
        assert len([x for x in test_files if x in val_files]) == 0
        assert len([x for x in val_files if x in train_files]) == 0

        if config["model"] == "loconlynet":
            # load in train dataset
            train_set = datastruc.LocDataset(
                None,  # raw_loc_dir_root
                train_folder,  # processed_dir_root
                label_level=config["label_level"],  # label_level
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
                range_xy=False,
            )

            # load in val dataset
            val_set = datastruc.LocDataset(
                None,  # raw_loc_dir_root
                val_folder,  # processed_dir_root
                label_level=config["label_level"],  # label_level
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
                range_xy=False,
            )

            # load in test dataset
            test_set = datastruc.LocDataset(
                None,  # raw_loc_dir_root
                test_folder,  # processed_dir_root
                label_level=config["label_level"],  # label_level
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
                range_xy=False,
            )

        elif config["model"] in [
            "locclusternet",
            "clusternet",
            "clustermlp",
            "locnetonly_pointnet",
            "locnetonly_pointtransformer",
        ]:
            train_set = datastruc.ClusterLocDataset(
                None,  # raw_loc_dir_root
                None,  # raw_cluster_dir_root
                train_folder,  # processed_dir_root
                label_level=config["label_level"],  # label_level
                pre_filter=None,  # pre_filter
                save_on_gpu=False,
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

            val_set = datastruc.ClusterLocDataset(
                None,  # raw_loc_dir_root
                None,  # raw_cluster_dir_root
                val_folder,  # processed_dir_root
                label_level=config["label_level"],  # label_level
                pre_filter=None,  # pre_filter
                save_on_gpu=False,
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

            test_set = datastruc.ClusterLocDataset(
                None,  # raw_loc_dir_root
                None,  # raw_cluster_dir_root
                test_folder,  # processed_dir_root
                label_level=config["label_level"],  # label_level
                pre_filter=None,  # pre_filter
                save_on_gpu=False,
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

        else:
            error_msg = config["model"]
            raise ValueError(f"Have not written for {error_msg}")

        train_loader = L.DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=0,
        )

        val_loader = L.DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=0,
        )

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
        model.load_state_dict(torch.load(model_loc))
        model.to(device)

        # model summary
        # print("\n")
        # print("---- Model summary ----")
        # print("\n")
        # number_nodes = 1000  # this is just for summary, has no bearing on training
        # summary(
        #    model,
        #    input_size=(test_set.num_node_features, number_nodes),
        #    batch_size=1,
        # )

        repeats = args.repeats

        # print("\n")
        # print("---- Predict on train set... ----")
        # print("\n")
        loc_dfs, cluster_dfs, fov_dfs, train_file_list = generate(
            gt_label_map,
            model,
            train_loader,
            device,
            fold,
            repeats=repeats,
        )

        loc_dfs = pl.concat(loc_dfs)
        train_loc_dfs.append(loc_dfs)

        cluster_dfs = pl.concat(cluster_dfs)
        train_cluster_dfs.append(cluster_dfs)

        fov_dfs = pl.concat(fov_dfs)
        train_fov_dfs.append(fov_dfs)

        # print("\n")
        # print("---- Predict on val set... ----")
        # print("\n")
        loc_dfs, cluster_dfs, fov_dfs, val_file_list = generate(
            gt_label_map,
            model,
            val_loader,
            device,
            fold,
            repeats=repeats,
        )

        loc_dfs = pl.concat(loc_dfs)
        val_loc_dfs.append(loc_dfs)

        cluster_dfs = pl.concat(cluster_dfs)
        val_cluster_dfs.append(cluster_dfs)

        fov_dfs = pl.concat(fov_dfs)
        val_fov_dfs.append(fov_dfs)

        # print("\n")
        # print("---- Predict on test set... ----")
        # print("\n")
        loc_dfs, cluster_dfs, fov_dfs, test_file_list = generate(
            gt_label_map,
            model,
            test_loader,
            device,
            fold,
            repeats=repeats,
        )

        loc_dfs = pl.concat(loc_dfs)
        test_loc_dfs.append(loc_dfs)

        cluster_dfs = pl.concat(cluster_dfs)
        test_cluster_dfs.append(cluster_dfs)

        fov_dfs = pl.concat(fov_dfs)
        test_fov_dfs.append(fov_dfs)

        OVERALL_test_file_list.extend(test_file_list)

        out = set(train_file_list) & set(val_file_list)
        assert not out
        out = set(train_file_list) & set(test_file_list)
        assert not out
        out = set(val_file_list) & set(test_file_list)
        assert not out

        print(f"{100*(fold + 1)/5}% complete")

    train_loc_dfs = pl.concat(train_loc_dfs)
    train_loc_path = os.path.join(project_directory, f"output/train_locs.csv")
    train_loc_dfs.write_csv(train_loc_path, separator=",")

    train_cluster_dfs = pl.concat(train_cluster_dfs)
    train_clusters_path = os.path.join(project_directory, f"output/train_clusters.csv")
    train_cluster_dfs.write_csv(train_clusters_path, separator=",")

    train_fov_dfs = pl.concat(train_fov_dfs)
    train_fov_path = os.path.join(project_directory, f"output/train_fovs.csv")
    train_fov_dfs.write_csv(train_fov_path, separator=",")

    val_loc_dfs = pl.concat(val_loc_dfs)
    val_loc_path = os.path.join(project_directory, f"output/val_locs.csv")
    val_loc_dfs.write_csv(val_loc_path, separator=",")

    val_cluster_dfs = pl.concat(val_cluster_dfs)
    val_clusters_path = os.path.join(project_directory, f"output/val_clusters.csv")
    val_cluster_dfs.write_csv(val_clusters_path, separator=",")

    val_fov_dfs = pl.concat(val_fov_dfs)
    val_fov_path = os.path.join(project_directory, f"output/val_fovs.csv")
    val_fov_dfs.write_csv(val_fov_path, separator=",")

    test_loc_dfs = pl.concat(test_loc_dfs)
    test_loc_path = os.path.join(project_directory, f"output/test_locs.csv")
    test_loc_dfs.write_csv(test_loc_path, separator=",")

    test_cluster_dfs = pl.concat(test_cluster_dfs)
    test_clusters_path = os.path.join(project_directory, f"output/test_clusters.csv")
    test_cluster_dfs.write_csv(test_clusters_path, separator=",")

    test_fov_dfs = pl.concat(test_fov_dfs)
    test_fov_path = os.path.join(project_directory, f"output/test_fovs.csv")
    test_fov_dfs.write_csv(test_fov_path, separator=",")

    return OVERALL_test_file_list


if __name__ == "__main__":
    main()
