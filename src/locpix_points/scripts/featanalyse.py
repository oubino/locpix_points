"""Feature analysis module

Module takes in the .parquet files and analyses features

Config file at top specifies the analyses we want to run
"""

import argparse
import json
import os
import time

from dig.xgraph.method import SubgraphX, GradCAM
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.evaluation import XCollector
from locpix_points.data_loading import datastruc
from locpix_points.models.cluster_nets import ClusterNetHomogeneous, parse_data
from locpix_points.models import model_choice
import matplotlib.colors as cl
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import torch
import torch_geometric.loader as L
from torch_geometric.explain import Explainer, AttentionExplainer, PGExplainer, metric
from torch_geometric.utils import to_undirected
import warnings
import yaml


class Present:
    """Required for visualising the positive and negative edges below"""

    def __init__(self):
        self.plot_present = [True, True, True, True]


def visualise_explanation(pos, edge_index, node_imp=None, edge_imp=None):
    """Visualise dataitem.
    Visualise the nodes with edges coloured by importance

    Args:
        pos (tensor) : Tensor containing node positions
        edge_index (tensor) : Tensor containing edge index
        node_imp (tensor) : Tensor denoting importance of each node from 0 to 1
        edge_imp (tensor) : Tensor denoting importance of each edge from 0 to 1
    """

    # edge_index and edge_imp to undirected
    warnings.warn(
        "The edge mask is directional, meaning an edge from node 0 to 1 may "
        "be significant while an edge from 1 to 0 may not be. "
        "For visualisation we make the graph undirected and we take the maximum"
        " of two edges that connect two nodes"
    )

    # if 1 or fewer important edges don't plot importance
    if edge_imp is not None and edge_imp.nonzero().shape[0] in [0, 1]:
        edge_imp = None
    if node_imp is not None and node_imp.nonzero().shape[0] in [0, 1]:
        node_imp = None

    plots = []

    # edge importance
    if edge_imp is not None:
        # make edges between nodes maximum of nodes connecting them
        edge_index, edge_imp = to_undirected(edge_index, edge_imp, reduce="max")
        # find nonzero/zero edges
        ind_nonzero = torch.squeeze(edge_imp.nonzero()).cpu().numpy()
        ind_zero = torch.squeeze((edge_imp == 0.0).nonzero()).cpu().numpy()

        pos = pos.cpu().numpy()
        edge_index = edge_index.cpu().numpy()
        # convert 2d to 3d if required
        if pos.shape[1] == 2:
            z = np.ones(pos.shape[0])
            z = np.expand_dims(z, axis=1)
            pos = np.concatenate([pos, z], axis=1)

        # color edges by importance
        lines = np.swapaxes(edge_index, 0, 1)
        colormap = get_cmap("seismic")
        rgba = colormap(edge_imp.cpu().numpy())
        colors = rgba[:, 0:3]

        # positive edges
        pos_edges = o3d.geometry.LineSet()
        pos_edges.points = o3d.utility.Vector3dVector(pos)
        pos_edges.lines = o3d.utility.Vector2iVector(lines[ind_nonzero, :])
        pos_edges.colors = o3d.utility.Vector3dVector(colors[ind_nonzero])

        # negative edges
        neg_edges = o3d.geometry.LineSet()
        neg_edges.points = o3d.utility.Vector3dVector(pos)
        neg_edges.lines = o3d.utility.Vector2iVector(lines[ind_zero, :])
        neg_edges.colors = o3d.utility.Vector3dVector(colors[ind_zero])

        plots.extend([pos_edges, neg_edges])

    else:
        pos = pos.cpu().numpy()
        edge_index = edge_index.cpu().numpy()
        # convert 2d to 3d if required
        if pos.shape[1] == 2:
            z = np.ones(pos.shape[0])
            z = np.expand_dims(z, axis=1)
            pos = np.concatenate([pos, z], axis=1)

        lines = np.swapaxes(edge_index, 0, 1)
        colors = np.full((len(lines), 3), fill_value=[0.33, 0.33, 0.33])
        edges = o3d.geometry.LineSet()
        edges.points = o3d.utility.Vector3dVector(pos)
        edges.lines = o3d.utility.Vector2iVector(lines)
        edges.colors = o3d.utility.Vector3dVector(colors)

        plots.append(edges)

    # node importance
    if node_imp is not None:
        # colors for nodes
        colormap = get_cmap("seismic")
        rgba = colormap(node_imp.cpu().numpy())
        colors = rgba[:, 0:3]

        # find nonzero/zero nodes
        ind_nonzero = torch.squeeze(node_imp.nonzero()).cpu().numpy()
        ind_zero = torch.squeeze((node_imp == 0.0).nonzero()).cpu().numpy()

        # positive nodes
        pos_nodes = o3d.geometry.PointCloud()
        pos_nodes.points = o3d.utility.Vector3dVector(pos[ind_nonzero])
        pos_nodes.colors = o3d.utility.Vector3dVector(colors[ind_nonzero])

        # negative nodes
        neg_nodes = o3d.geometry.PointCloud()
        if type(ind_zero) == list:
            pos = pos[ind_zero]
            colors = colors[ind_zero]
        else:
            pos = [pos[ind_zero]]
            colors = [colors[ind_zero]]
        neg_nodes.points = o3d.utility.Vector3dVector(pos)
        neg_nodes.colors = o3d.utility.Vector3dVector(colors)

        plots.extend([pos_nodes, neg_nodes])

    else:
        colors = np.full((len(pos), 3), fill_value=[0.33, 0.33, 0.33])
        nodes = o3d.geometry.PointCloud()
        nodes.points = o3d.utility.Vector3dVector(pos)
        nodes.colors = o3d.utility.Vector3dVector(colors)

        plots.append(nodes)

    # add key callbacks
    present = Present()

    def visualise_plot_zero(vis):
        """Function needed for key binding to visualise first plot

        Args:
            vis (o3d.Visualizer): Visualizer to load/remove data from"""
        if present.plot_present[0]:
            vis.remove_geometry(plots[0], False)
            present.plot_present[0] = False
        else:
            vis.add_geometry(plots[0], False)
            present.plot_present[0] = True

    def visualise_plot_one(vis):
        """Function needed for key binding to visualise second plot

        Args:
            vis (o3d.Visualizer): Visualizer to load/remove data from"""
        if present.plot_present[1]:
            vis.remove_geometry(plots[1], False)
            present.plot_present[1] = False
        else:
            vis.add_geometry(plots[1], False)
            present.plot_present[1] = True

    def visualise_plot_two(vis):
        """Function needed for key binding to visualise third plot

        Args:
            vis (o3d.Visualizer): Visualizer to load/remove data from"""
        if present.plot_present[2]:
            vis.remove_geometry(plots[2], False)
            present.plot_present[2] = False
        else:
            vis.add_geometry(plots[2], False)
            present.plot_present[2] = True

    def visualise_plot_three(vis):
        """Function needed for key binding to visualise fourth plot

        Args:
            vis (o3d.Visualizer): Visualizer to load/remove data from"""
        if present.plot_present[3]:
            vis.remove_geometry(plots[3], False)
            present.plot_present[3] = False
        else:
            vis.add_geometry(plots[3], False)
            present.plot_present[3] = True

    key_to_callback = {}

    if edge_imp is None and node_imp is None:
        key_to_callback[ord("R")] = visualise_plot_zero
        key_to_callback[ord("K")] = visualise_plot_one
        print("Add/Remove edges using R button")
        print("Add/Remove nodes using K button")

    elif edge_imp is not None and node_imp is None:
        key_to_callback[ord("R")] = visualise_plot_zero
        key_to_callback[ord("K")] = visualise_plot_one
        key_to_callback[ord("T")] = visualise_plot_two
        print("Add/Remove positive edges using R button")
        print("Add/Remove negative edges using K button")
        print("Add/Remove nodes using T button")

    elif edge_imp is None and node_imp is not None:
        key_to_callback[ord("R")] = visualise_plot_zero
        key_to_callback[ord("T")] = visualise_plot_one
        key_to_callback[ord("Y")] = visualise_plot_two
        print("Add/Remove edges using R button")
        print("Add/Remove positive nodes using T button")
        print("Add/Remove negative nodes using Y button")

    elif edge_imp is not None and node_imp is not None:
        key_to_callback[ord("R")] = visualise_plot_zero
        key_to_callback[ord("K")] = visualise_plot_one
        key_to_callback[ord("T")] = visualise_plot_two
        key_to_callback[ord("Y")] = visualise_plot_three
        print("Add/Remove positive edges using R button")
        print("Add/Remove negative edges using K button")
        print("Add/Remove positive nodes using T button")
        print("Add/Remove negative nodes using Y button")

    _ = o3d.visualization.Visualizer()
    o3d.visualization.draw_geometries_with_key_callbacks(plots, key_to_callback)


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If no files present to open"""

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
        "-n",
        "--neuralnet",
        action="store_true",
        help="if present then the output of the neural"
        "net is analysed rather than the manual features",
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

    # list items
    try:
        if not args.final_test:
            train_loc_files = os.listdir(
                os.path.join(project_directory, "preprocessed/featextract/locs")
            )
            test_loc_files = None
        else:
            train_loc_files = os.listdir(
                os.path.join(project_directory, "preprocessed/train/featextract/locs")
            )
            test_loc_files = os.listdir(
                os.path.join(project_directory, "preprocessed/test/featextract/locs")
            )
    except FileNotFoundError:
        raise ValueError("There should be some loc files to open")

    try:
        if not args.final_test:
            train_cluster_files = os.listdir(
                os.path.join(project_directory, "preprocessed/featextract/clusters")
            )
            test_cluster_files = None
        else:
            train_cluster_files = os.listdir(
                os.path.join(
                    project_directory, "preprocessed/train/featextract/clusters"  # edit
                )
            )
            test_cluster_files = os.listdir(
                os.path.join(
                    project_directory, "preprocessed/test/featextract/clusters"  # edit
                )
            )
    except FileNotFoundError:
        raise ValueError("There should be some cluster files to open")

    assert train_loc_files == train_cluster_files
    assert test_loc_files == test_cluster_files

    # make seaborn plots pretty
    # sns.set_style("darkgrid")

    # make output folder
    output_folder = os.path.join(project_directory, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- Analyse cluster features -------
    if not args.neuralnet:
        analyse_manual_feats(project_directory, train_loc_files, test_loc_files, args)
    elif args.neuralnet:
        analyse_nn_feats(project_directory, config, args)
    else:
        raise ValueError("Should be neural net or manual")


def analyse_manual_feats(
    project_directory,
    train_loc_files,
    test_loc_files,
    args,
):
    """Analyse the features of the clusters manually extracted

    Args:
        project_directory (str): Location of the project directory
        train_loc_files (list): List of the TRAIN files with the protein
            localisations
        test_loc_files (list): List of the TEST files with the protein
            localisations
        args (dict): Arguments passed to this script
    """

    # aggregate cluster features into collated df
    train_dfs = []

    if not args.final_test:
        train_cluster_root = os.path.join(
            project_directory, f"preprocessed/featextract/clusters"
        )
    else:
        train_cluster_root = os.path.join(
            project_directory, f"preprocessed/train/featextract/clusters"
        )
        # process test data
        test_cluster_root = os.path.join(
            project_directory, f"preprocessed/test/featextract/clusters"
        )
        test_dfs = []
        for index, file in enumerate(test_loc_files):
            test_cluster_path = os.path.join(test_cluster_root, file)

            cluster_df = pq.read_table(test_cluster_path)
            # extract metadata
            gt_label_map = json.loads(
                cluster_df.schema.metadata[b"gt_label_map"].decode("utf-8")
            )
            gt_label_map = {int(key): value for key, value in gt_label_map.items()}
            gt_label = cluster_df.schema.metadata[b"gt_label"]
            gt_label = int(gt_label)
            label = gt_label_map[gt_label]

            # convert to polars
            cluster_df = pl.from_arrow(cluster_df)
            cluster_df = cluster_df.with_columns(pl.lit(label).alias("type"))
            cluster_df = cluster_df.with_columns(pl.lit(f"{file}").alias("file_name"))
            test_dfs.append(cluster_df)

    # process training data
    for index, file in enumerate(train_loc_files):
        train_cluster_path = os.path.join(train_cluster_root, file)

        cluster_df = pq.read_table(train_cluster_path)

        # extract metadata
        gt_label_map = json.loads(
            cluster_df.schema.metadata[b"gt_label_map"].decode("utf-8")
        )
        gt_label_map = {int(key): value for key, value in gt_label_map.items()}
        gt_label = cluster_df.schema.metadata[b"gt_label"]
        gt_label = int(gt_label)
        label = gt_label_map[gt_label]

        # convert to polars
        cluster_df = pl.from_arrow(cluster_df)
        cluster_df = cluster_df.with_columns(pl.lit(label).alias("type"))
        cluster_df = cluster_df.with_columns(pl.lit(f"{file}").alias("file_name"))
        train_dfs.append(cluster_df)

    # aggregate dfs into one big df
    train_df = pl.concat(train_dfs)
    train_df = train_df.to_pandas()
    if args.final_test:
        test_df = pl.concat(test_dfs)
        test_df = test_df.to_pandas()

    # save train and test df
    train_df.to_csv(
        os.path.join(project_directory, "output/train_df_manual.csv"), index=False
    )
    if args.final_test:
        test_df.to_csv(
            os.path.join(project_directory, "output/test_df_manual.csv"), index=False
        )


def analyse_nn_feats(project_directory, config, args):
    """Analyse the features of the clusters from neural network

    Args:
        project_directory (str): Location of the project directory
        config (dict): Configuration for this script
        args (dict): Arguments passed to this script

    Raises:
        ValueError: If device specified is neither cpu or gpu OR
            if attention to examine is not correctly specified OR
            if encoder not loc or cluster or fov
        NotImplementedError: If try to run attention on Loc or
            LocCluster instead of cluster
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
    # 1. To construct datasets we use cluster_net required in model
    # 2. For explainability also assumes uses LocClusterNet
    assert model_type == "locclusternet"

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

    # need to create a homogenous dataset consisting only of clusters from the heterogeneous graph
    data_folder = os.path.join(project_directory, "processed", "featanalysis")

    if not args.final_test:
        input_train_folder = os.path.join(
            project_directory, "processed", f"fold_{fold}", "train"
        )
        input_val_folder = os.path.join(
            project_directory, "processed", f"fold_{fold}", "val"
        )
        input_test_folder = os.path.join(
            project_directory, "processed", f"fold_{fold}", "test"
        )
    else:
        input_train_folder = os.path.join(project_directory, "processed", "train")
        input_val_folder = os.path.join(project_directory, "processed", "val")
        input_test_folder = os.path.join(project_directory, "processed", "test")
    output_train_folder = os.path.join(data_folder, "train")
    output_val_folder = os.path.join(data_folder, "val")
    output_test_folder = os.path.join(data_folder, "test")

    output_folders = [output_train_folder, output_val_folder, output_test_folder]
    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # initialise train/validation and test sets

    # note these datastructures have been passed through
    # PointNet/PointTransformer therefore localisations
    # have been embedded into cluster
    loc_model = model.loc_net
    loc_model.eval()

    cluster_train_set = datastruc.ClusterDataset(
        input_train_folder,
        output_train_folder,
        label_level=None,
        pre_filter=None,
        save_on_gpu=None,
        transform=None,
        pre_transform=None,
        fov_x=None,
        fov_y=None,
        from_hetero_loc_cluster=True,
        loc_net=loc_model,
        device=device,
    )

    cluster_val_set = datastruc.ClusterDataset(
        input_val_folder,
        output_val_folder,
        label_level=None,
        pre_filter=None,
        save_on_gpu=None,
        transform=None,
        pre_transform=None,
        fov_x=None,
        fov_y=None,
        from_hetero_loc_cluster=True,
        loc_net=loc_model,
        device=device,
    )

    cluster_test_set = datastruc.ClusterDataset(
        input_test_folder,
        output_test_folder,
        label_level=None,
        pre_filter=None,
        save_on_gpu=None,
        transform=None,
        pre_transform=None,
        fov_x=None,
        fov_y=None,
        from_hetero_loc_cluster=True,
        loc_net=loc_model,
        device=device,
    )

    # test set where localisations are yet to be passed through any network

    # load in test dataset
    loc_test_set = datastruc.ClusterLocDataset(
        None,  # raw_loc_dir_root
        None,  # raw_cluster_dir_root
        input_test_folder,  # processed_dir_root
        label_level=None,
        pre_filter=None,
        save_on_gpu=None,
        transform=None,
        pre_transform=None,
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

    # ------- GRAPHXAI --------

    warnings.warn(
        "Assumes\
                      1. Using a ClusterNet \
                      2. Embedding is in a final cluster encoder layer\
                      3. Graph level explanation"
    )

    # need to create a model that acts on the homogeneous data for cluster and locs
    cluster_model = ClusterNetHomogeneous(model.cluster_net, config[model_type])
    cluster_model.to(device)
    cluster_model.eval()

    # train pgexplainer
    if "pgex" in config.keys():
        print("Training PGEX...")
        max_epochs = config["pgex"]["max_epochs"]
        lr = config["pgex"]["lr"]
        edge_size = config["pgex"]["edge_size"]
        edge_ent = config["pgex"]["edge_ent"]
        temp = config["pgex"]["temp"]
        bias = config["pgex"]["bias"]

        # PGExplainer make it return logprobs
        pg_explainer = Explainer(
            model=cluster_model,
            algorithm=PGExplainer(
                epochs=max_epochs,
                lr=lr,
                edge_size=edge_size,
                edge_ent=edge_ent,
                temp=temp,
                bias=bias,
            ),
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="graph",
                return_type="log_probs",
            ),
        )
        pg_explainer.algorithm.mlp.to(device)

        ## Required to first train PGExplainer on the dataset:
        pgex_train_set = torch.utils.data.ConcatDataset(
            [cluster_train_set, cluster_val_set]
        )

        batch_size = config["pgex"]["batch_size"]

        # initialise loader
        train_loader = L.DataLoader(
            pgex_train_set,
            batch_size=batch_size,  # change in config
            shuffle=True,
            drop_last=True,
        )

        # train pgexplainer
        print("Training")
        for epoch in range(max_epochs):
            total_loss = 0
            items = 0
            for index, item in enumerate(train_loader):
                loss = pg_explainer.algorithm.train(
                    epoch,
                    cluster_model,
                    item.x,
                    item.edge_index,
                    target=item.y,
                    pos=item.pos,
                    batch=item.batch,
                    # return logprobs
                    logits=False,
                )
                items += index * batch_size
                total_loss += loss
            print(f"Epoch: {epoch}; Loss : {total_loss/items}")

    # get item to evaluate on
    dataitem_idx = config["dataitem"]
    for idx in dataitem_idx:
        cluster_dataitem = cluster_test_set.get(idx)
        loc_dataitem = loc_test_set.get(idx)
        loc_dataitem.to(device)

        # generate prediction for the graph
        logits = cluster_model(
            cluster_dataitem.x,
            cluster_dataitem.edge_index,
            torch.tensor([0], device=device),
            cluster_dataitem.pos,
            logits=True,
        )
        prediction = logits.argmax(-1).item()

        # print out prediction & gt label
        print("-----")
        print(f"Item {idx}")
        print("Predicted label: ", gt_label_map[prediction])
        print("GT label: ", gt_label_map[cluster_dataitem.y.cpu().item()])

        # ---- subgraphx -----
        if "subgraphx" in config.keys():
            print("Subgraphx...")
            explainer = SubgraphX(
                cluster_model,
                num_classes=config["subgraphx"]["num_classes"],
                device=device,
                explain_graph=True,
                rollout=config["subgraphx"]["rollout"],
                min_atoms=config["subgraphx"]["min_atoms"],
                c_puct=config["subgraphx"]["c_puct"],
                expand_atoms=config["subgraphx"]["expand_atoms"],
                high2low=config["subgraphx"]["high2low"],
                local_radius=config["subgraphx"]["local_radius"],
                sample_num=config["subgraphx"]["sample_num"],
                reward_method=config["subgraphx"]["reward_method"],
                subgraph_building_method=config["subgraphx"][
                    "subgraph_building_method"
                ],
                vis=False,
            )

            # generate explanation for the graph
            _, explanation_results, related_preds = explainer(
                cluster_dataitem.x,
                cluster_dataitem.edge_index,
                forward_kwargs={
                    "batch": torch.tensor([0], device=device),
                    "pos": cluster_dataitem.pos,
                    "logits": True,  # has to be True
                },
                max_nodes=config["subgraphx"]["max_nodes"],
            )

            # process explanation results
            explanation_results = explanation_results[prediction]
            explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)
            tree_node_x = find_closest_node_result(
                explanation_results, max_nodes=config["subgraphx"]["max_nodes"]
            )

            # generate metrics for explanation
            nodelist = tree_node_x.coalition
            node_imp = torch.zeros(len(cluster_dataitem.pos))
            node_imp[nodelist] = 1.0
            x_collector = XCollector()
            x_collector.collect_data(
                tree_node_x.coalition, related_preds, label=prediction
            )

            # print metrics for explanation
            print(f"Positive fidelity closer to 1 better: {x_collector.fidelity:.4f})")
            print(
                f"Negative fidelity closer to 0 better: {x_collector.fidelity_inv:.4f})"
            )
            print(f"Sparsity: {x_collector.sparsity:.4f}")
            print(f"Accuracy: {x_collector.accuracy:.4f}")
            print(f"Stability: {x_collector.stability:.4f}")

            # evaluate explanation

            visualise_explanation(
                cluster_dataitem.pos,
                cluster_dataitem.edge_index,
                node_imp=node_imp.to(device),
                edge_imp=None,
            )

        # ---- gradcam ----
        if "gradcam" in config.keys():
            raise NotImplementedError("GradCAM not implemented yet")
            print("GradCAM...")
            # To implement GradCAM we need to remove all MLP layers
            # as gradcam works by getting the weight attribute of the layer
            # but MLP doesn't have this attribute therefore replace all MLP
            # with the parts of it
            explainer = GradCAM(
                cluster_model,
                explain_graph=True,
            )

            # generate explanation for the graph
            edge_masks, hard_edge_masks, related_preds = explainer(
                cluster_dataitem.x,
                cluster_dataitem.edge_index,
                forward_kwargs={
                    "batch": torch.tensor([0], device=device),
                    "pos": cluster_dataitem.pos,
                    "logits": True,  # has to be True
                },
                # max_nodes=config["gradcam"]["max_nodes"],
                num_classes=config["gradcam"]["num_classes"],
                sparsity=config["gradcam"]["sparsity"],
            )

            # generate metrics for explanation
            x_collector = XCollector()
            x_collector.collect_data(
                hard_edge_masks, related_preds, label=cluster_dataitem.y
            )

            # print metrics for explanation
            print(f"Positive fidelity closer to 1 better: {x_collector.fidelity:.4f})")
            print(
                f"Negative fidelity closer to 0 better: {x_collector.fidelity_inv:.4f})"
            )
            print(f"Sparsity: {x_collector.sparsity:.4f}")
            print(f"Accuracy: {x_collector.accuracy:.4f}")
            print(f"Stability: {x_collector.stability:.4f}")

            # evaluate explanation
            visualise_explanation(
                cluster_dataitem.pos,
                cluster_dataitem.edge_index,
                node_imp=node_imp.to(device),
                edge_imp=None,
            )

        # ---- pgexplainer ----
        if "pgex" in config.keys():
            print("PGEX evaluate...")

            # explain cluster dataitem
            explanation = pg_explainer(
                cluster_dataitem.x,
                cluster_dataitem.edge_index,
                target=cluster_dataitem.y,
                pos=cluster_dataitem.pos,
                batch=torch.zeros(
                    cluster_dataitem.x.shape[0], device=device, dtype=torch.int64
                ),
                # return logprobs
                logits=False,
            )
            # metrics
            print(
                f"Warning there are {torch.count_nonzero(explanation.edge_mask)} non zero elements in the edge mask out of {len(explanation.edge_mask)} elements"
            )
            explanation.edge_mask = torch.where(
                explanation.edge_mask > config["edge_mask_threshold"],
                explanation.edge_mask,
                0.0,
            )
            print(
                f"Post thresholding there are {torch.count_nonzero(explanation.edge_mask)} non zero elements in the edge mask out of {len(explanation.edge_mask)} elements"
            )
            pos_fid, neg_fid = metric.fidelity(pg_explainer, explanation)
            print(f"Positive fidelity closer to 1 better: {pos_fid})")
            print(f"Negative fidelity closer to 0 better: {neg_fid})")
            unf = metric.unfaithfulness(pg_explainer, explanation)
            print(f"Unfaithfulness, closer to 0 better {unf}")

            # visualise explanation
            visualise_explanation(
                cluster_dataitem.pos,
                cluster_dataitem.edge_index,
                node_imp=None,
                edge_imp=explanation.edge_mask.to(device),
            )

        # ---- attention -----
        # use model - logprobs or clustermodel - raw
        if "attention" in config.keys():
            print("---- Attention ----")
            scale = config["attention"]["scale"]
            reduce = config["attention"]["reduce"]

            if scale == "cluster":
                attention_model = cluster_model
                return_type = "log_probs"  # attention doesn't use the target nor the return type which is used to generate the target therefore this argument is irrelevant
            elif scale == "loc":
                raise NotImplementedError(
                    "Not currently implementing attention for PointTransformer as not easy to implement 1. Edges not fixed constructed on the go by Transformer 2. Input to forward method is data not x, edge_index etc"
                )
                attention_model = loc_model
                return_type = "log_probs"  # attention doesn't use the target nor the return type which is used to generate the target therefore this argument is irrelevant
            elif scale == "loccluster":
                raise NotImplementedError(
                    "LocCluster is not currently implemented as attention explainer doesn't accept hereogeneous graphs"
                )
            else:
                raise NotImplementedError(
                    "attention model must be cluster, point or pointcluster"
                )

            explainer = Explainer(
                model=attention_model,
                algorithm=AttentionExplainer(reduce=reduce),
                explanation_type="model",
                node_mask_type=None,
                edge_mask_type="object",
                model_config=dict(
                    mode="multiclass_classification",
                    task_level="graph",
                    # return logprobs
                    return_type=return_type,
                ),
            )

            if scale == "cluster":
                explanation = explainer(
                    x=cluster_dataitem.x,
                    edge_index=cluster_dataitem.edge_index,
                    target=None,  # attention doesn't use the target nor the return type which is used to generate the target therefore this argument is irrelevant
                    batch=torch.tensor([0], device=device),
                    pos=cluster_dataitem.pos,
                    # return logprobs
                    logits=False,
                )
                # metrics
                print(
                    f"Warning there are {torch.count_nonzero(explanation.edge_mask)} non zero elements in the edge mask out of {len(explanation.edge_mask)} elements"
                )
                explanation.edge_mask = torch.where(
                    explanation.edge_mask > config["edge_mask_threshold"],
                    explanation.edge_mask,
                    0.0,
                )
                print(
                    f"Post thresholding there are {torch.count_nonzero(explanation.edge_mask)} non zero elements in the edge mask out of {len(explanation.edge_mask)} elements"
                )
                pos_fid, neg_fid = metric.fidelity(explainer, explanation)
                print(f"Positive fidelity closer to 1 better: {pos_fid})")
                print(f"Negative fidelity closer to 0 better: {neg_fid})")
                unf = metric.unfaithfulness(explainer, explanation)
                print(f"Unfaithfulness, closer to 0 better {unf}")
                visualise_explanation(
                    cluster_dataitem.pos,
                    cluster_dataitem.edge_index,
                    node_imp=None,
                    edge_imp=explanation.edge_mask.to(device),
                )
            elif scale == "loc":
                loc_x_dict, loc_pos_dict, loc_edge_index_dict, _ = parse_data(
                    loc_dataitem, None
                )
                explanation = explainer(
                    x=loc_x_dict["locs"],
                    edge_index=loc_edge_index_dict["locs", "in", "clusters"],
                    pos_locs=loc_pos_dict["locs"],
                    target=None,  # attention doesn't use the target nor the return type which is used to generate the target therefore this argument is irrelevant
                    # batch=torch.tensor([0], device=device),
                    logits=False,
                )
                # metrics
                print(
                    f"Warning there are {torch.count_nonzero(explanation.edge_mask)} non zero elements in the edge mask out of {len(explanation.edge_mask)} elements"
                )
                explanation.edge_mask = torch.where(
                    explanation.edge_mask > config["edge_mask_threshold"],
                    explanation.edge_mask,
                    0.0,
                )
                print(
                    f"Post thresholding there are {torch.count_nonzero(explanation.edge_mask)} non zero elements in the edge mask out of {len(explanation.edge_mask)} elements"
                )
                pos_fid, neg_fid = metric.fidelity(explainer, explanation)
                print(f"Positive fidelity closer to 1 better: {pos_fid})")
                print(f"Negative fidelity closer to 0 better: {neg_fid})")
                unf = metric.unfaithfulness(explainer, explanation)
                print(f"Unfaithfulness, closer to 0 better {unf}")
                visualise_explanation(
                    loc_pos_dict["locs"],
                    loc_edge_index_dict["locs", "in", "clusters"],
                    node_imp=None,
                    edge_imp=explanation.edge_mask.to(device),
                )

    # ------- BOXPLOT/UMAP/SKLEARN SETUP ---------

    print("Feature analysis...")

    # need to ensure no manual features being analysed
    with open("config/process.yaml", "r") as ymlfile:
        process_config = yaml.safe_load(ymlfile)
    cluster_features = process_config["cluster_feat"]
    assert cluster_features is None

    # aggregate cluster features into collated df
    if not args.final_test:
        train_dataset = torch.utils.data.ConcatDataset(
            [cluster_train_set, cluster_val_set, cluster_test_set]
        )
        test_dataset = None
    else:
        train_dataset = torch.utils.data.ConcatDataset(
            [cluster_train_set, cluster_val_set]
        )
        test_dataset = cluster_test_set

    features_to_csv(
        train_dataset,
        test_dataset,
        cluster_model,
        gt_label_map,
        "loc",
        device,
        project_directory,
        args.final_test,
    )
    features_to_csv(
        train_dataset,
        test_dataset,
        cluster_model,
        gt_label_map,
        "cluster",
        device,
        project_directory,
        args.final_test,
    )
    features_to_csv(
        train_dataset,
        test_dataset,
        cluster_model,
        gt_label_map,
        "fov",
        device,
        project_directory,
        args.final_test,
    )


def features_to_csv(
    train_dataset,
    test_dataset,
    cluster_model,
    gt_label_map,
    encoder,
    device,
    project_directory,
    final_test,
):
    """Convert features to .csv file

    Args:
        train_dataset (dataset): Training dataset
        test_dataset (dataset): Test dataset
        cluster_model (pyg model): PyG model
        gt_label_map (dict): GT label map
        encoder (string): Which encoder to use
        device (torch deive): What device its on
        project_directory (string): Where is project directory
        final_test (bool): Whether is final test or not"""

    train_dfs = get_features(
        train_dataset, cluster_model, gt_label_map, encoder, device
    )
    if final_test:
        test_dfs = get_features(
            test_dataset, cluster_model, gt_label_map, encoder, device
        )

    # aggregate dfs into one big df
    train_df = pl.concat(train_dfs)
    train_df = train_df.to_pandas()
    if final_test:
        test_df = pl.concat(test_dfs)
        test_df = test_df.to_pandas()

    # save train and test df
    train_df.to_csv(
        os.path.join(project_directory, f"output/train_df_nn_{encoder}.csv"),
        index=False,
    )
    if final_test:
        test_df.to_csv(
            os.path.join(project_directory, f"output/test_df_nn_{encoder}.csv"),
            index=False,
        )


def get_features(dataset, cluster_model, gt_label_map, encoder, device):
    """Get features from the neural network

    Args:
        dataset (dataset): PyG dataset to get features for
        cluster_model (model): PyG model to get features out of
        gt_label_map (dict): Dictionary mapping gt labels
        encoder (string): Either loc, cluster or fov
        device (torch device): Device running on

    Returns:
        dfs (list): Dataframes with features from encoding

    Raises:
        ValueError: If encoder not loc, fov or cluster
    """

    dfs = []

    # a dict to store the activations
    activation = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # register forward hook
    h_0 = cluster_model.cluster_encoder_3.register_forward_hook(
        getActivation("clusterencoder")
    )
    h_1 = cluster_model.pool.register_forward_hook(getActivation("globalpool"))

    print("Encoder: ", encoder)
    predictions = []
    labels = []

    for _, data in enumerate(dataset):
        # gt label
        gt_label = int(data.y)
        label = gt_label_map[gt_label]

        # file name
        file_name = data.name + ".parquet"

        # forward through network
        prediction = cluster_model(
            data.x,
            data.edge_index,
            torch.tensor([0], device=device),
            data.pos,
            logits=True,
        )

        predictions.append(prediction.log_softmax(dim=-1).argmax().cpu().item())
        labels.append(data.y[0].cpu().item())

        # convert to polars
        if encoder == "loc":
            data = data.x.detach().cpu().numpy()
        elif encoder == "cluster":
            data = activation["clusterencoder"].cpu().numpy()
        elif encoder == "fov":
            data = activation["globalpool"].cpu().numpy()
        else:
            raise ValueError("encoder should be loc or cluster")
        cluster_df = pl.DataFrame(data)
        cluster_df = cluster_df.with_columns(pl.lit(label).alias("type"))
        cluster_df = cluster_df.with_columns(pl.lit(f"{file_name}").alias("file_name"))
        dfs.append(cluster_df)

    print("accuracy: ", accuracy_score(labels, predictions))

    # remove foward hook
    h_0.remove()
    h_1.remove()

    return dfs


# save yaml file
# yaml_save_loc = os.path.join(project_directory, "featextract.yaml")
# with open(yaml_save_loc, "w") as outfile:
#    yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
