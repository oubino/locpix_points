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
import warnings

import torch
import torch_geometric.loader as L
import yaml
from graphxai.explainers import PGExplainer
from torchsummary import summary

import wandb
from locpix_points.data_loading import datastruc
from locpix_points.evaluation import evaluate
from locpix_points.models import model_choice
from locpix_points.models.cluster_nets import ClusterNetHomogeneous

# import torch
# import torch_geometric.transforms as T


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If GPU/CPU argument incorrectly specified
        NotImplementedError: If model/explainer not implemented"""

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

    parser.add_argument("-e", "--explain", action="store_true")

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
        raise ValueError(
            "No gpu available, can run on cpu\
                         instead"
        )
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

    test_loader = L.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
    )
    for _, data in enumerate(test_loader):
        first_item = data
    dim = first_item["locs"].pos.shape[-1]
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
    model.load_state_dict(torch.load(args.model_loc))
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

    # explain
    if args.explain:
        warnings.warn(
            "Assumes\
                      1. Using a ClusterNet \
                      2. Embedding is in a final cluster encoder layer\
                      3. Graph level explanation"
        )

        # load in explain params
        mult = 2  # in pg_explainer mult is 2 unless it is per node then mult is 3
        in_channels = config[config["model"]]["ClusterEncoderChannels"][-1][-1]
        in_channels *= mult

        # need to create a homogenous dataset consisting only of clusters from the heterogeneous graph
        explain_folder = os.path.join(processed_directory, "explain")
        if not os.path.exists(explain_folder):
            os.makedirs(explain_folder)
        explain_dataset = datastruc.ClusterDataset(
            test_folder,
            explain_folder,
            label_level=None,
            pre_filter=None,
            save_on_gpu=None,
            transform=None,
            pre_transform=None,
            fov_x=None,
            fov_y=None,
            from_hetero_loc_cluster=True,
            loc_net=model.loc_net,
            device=device,
        )

        # need to create a model that acts on the homogeneous data
        if config["model"] == "locclusternet":
            model = ClusterNetHomogeneous(model.cluster_net, config["locclusternet"])
        else:
            raise NotImplementedError

        print("\n")
        print("---- Predict on test set (in explain)... ----")
        explain_loader = L.DataLoader(
            explain_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )
        print("\n")
        metrics = evaluate.make_prediction_test(
            model,
            explain_loader,
            device,
            num_classes,
            explain=True,
        )

        for key, value in metrics.items():
            print(f"{key} : {value.item()}")

        if "pgex" in config.keys():
            emb_layer_name = config["pgex"]["emb_layer_name"]
            coeff_size = config["pgex"]["coeff_size"]
            coeff_ent = config["pgex"]["coeff_ent"]
            explain_graph = config["pgex"]["explain_graph"]
            max_epochs = config["pgex"]["max_epochs"]
            lr = config["pgex"]["lr"]

            # Embedding layer name is final GNN embedding layer in the model
            # Note that PGExplainer expects unnormalized class score as final output
            # therefore final layer is linear
            pgex = PGExplainer(
                model,
                emb_layer_name=emb_layer_name,
                coeff_size=coeff_size,
                coeff_ent=coeff_ent,
                explain_graph=explain_graph,
                max_epochs=max_epochs,
                lr=lr,
                in_channels=in_channels,
            )

            # Required to first train PGExplainer on the dataset:
            pgex.train_explanation_model(
                explain_dataset,
                forward_kwargs={"batch": torch.tensor([0], device=device)},
            )

            # Get explanations from both IG and PGEx:
            for index, data in enumerate(explain_dataset):
                last_item = data

            pgex_exp = pgex.get_explanation_graph(
                x=last_item.x,
                edge_index=last_item.edge_index,
                label=last_item.y,
                forward_kwargs={"batch": torch.tensor([0], device=device)},
            )

            print("Feature importance")
            print(pgex_exp.feature_imp)
            print("Node importance")
            imp = pgex_exp.node_imp
            print(imp)
            print("Edge importance")
            imp = pgex_exp.edge_imp
            print(imp)

            # fig, ax = plt.subplots(1,3, figsize = (10, 8))

            # Ground-truth explanations always provided as a list. In ShapeGGen, we use the first
            #   element since it produces unique explanations.
            # pgex_exp.visualize_graph(ax = ax[1],
            #                         show=True)

        else:
            raise NotImplementedError
    else:
        print("\n")
        print("---- Predict on test set... ----")
        print("\n")
        metrics = evaluate.make_prediction_test(
            model,
            test_loader,
            device,
            num_classes,
            explain=False,
        )

        for key, value in metrics.items():
            print(f"{key} : {value.item()}")

    wandb.log(metrics)

    time_o = time.gmtime(time.time())
    time_o = f"{time_o[3]}:{time_o[4]}_{time_o[2]}:{time_o[1]}:{time_o[0]}"

    # save config file to folder and wandb
    yaml_save_loc = os.path.join(project_directory, f"evaluate_{time_o}.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)
    yaml_save_loc = os.path.join(wandb.run.dir, f"evaluate_{time_o}.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)

    # exit wandb
    if not args.wandbstarted:
        wandb.finish()


if __name__ == "__main__":
    main()
