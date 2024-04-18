"""Feature analysis module

Module takes in the .parquet files and analyses features

Config file at top specifies the analyses we want to run
"""

import argparse
import json
import os
import time

from graphxai.explainers import SubgraphX, GuidedBP, GradCAM
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import torch
import torch_geometric.loader as L
from torch_geometric.explain import Explainer, AttentionExplainer, PGExplainer, metric
import warnings
import yaml


def visualise_cluster_explanation(dataitem, node_imp, edge_imp):
    """Visualise dataitem.
    Visualise the clusters coloured by importance

    Args:
        dataitem (torch geometric dataitem) : Pytorch geometric data item to visualise
        node_imp (numpy array) : Numpy array of 1.0 or 0.0 according to if node is important
            or not
        edge_imp (numpy array) : Numpy array of 1.0 or 0.0 according to if edge is important
            or not"""

    pos = dataitem.pos.cpu().numpy()
    edge_index = dataitem.edge_index.cpu().numpy()

    # convert 2d to 3d if required
    if pos.shape[1] == 2:
        z = np.ones(pos.shape[0])
        z = np.expand_dims(z, axis=1)
        pos = np.concatenate([pos, z], axis=1)

    # cluster to cluster edges
    lines = np.swapaxes(edge_index, 0, 1)
    colors = np.full((len(lines), 3), fill_value=[0.33, 0.33, 0.33])
    idx = edge_imp.nonzero()
    colors[idx] = [1, 1, 0]

    clusters_to_clusters = o3d.geometry.LineSet()
    clusters_to_clusters.points = o3d.utility.Vector3dVector(pos)
    clusters_to_clusters.lines = o3d.utility.Vector2iVector(lines)
    clusters_to_clusters.colors = o3d.utility.Vector3dVector(colors)

    # cluster positions
    colors = np.full((len(pos), 3), fill_value=[0.33, 0.33, 0.33])
    idx = node_imp.nonzero()
    colors[idx] = [1, 0, 0]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pos)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # visualise
    _ = o3d.visualization.Visualizer()
    o3d.visualization.draw_geometries([point_cloud, clusters_to_clusters])


def visualise_edge_mask(pos, edge_index, edge_mask):
    """Visualise dataitem.
    Visualise the nodes with edges coloured by importance

    Args:
        pos (tensor) : Tensor containing node positions
        edge_index (tensor) : Tensor containing edge index
        edge_mask (numpy array) : Numpy array denoting importance of each edge from 0 to 1
    """

    pos = pos.cpu().numpy()
    edge_index = edge_index.cpu().numpy()

    # convert 2d to 3d if required
    if pos.shape[1] == 2:
        z = np.ones(pos.shape[0])
        z = np.expand_dims(z, axis=1)
        pos = np.concatenate([pos, z], axis=1)

    # node to node edges
    lines = np.swapaxes(edge_index, 0, 1)
    colormap = get_cmap("seismic")
    rgba = colormap(edge_mask.cpu().numpy())
    colors = rgba[:, 0:3]

    edges = o3d.geometry.LineSet()
    edges.points = o3d.utility.Vector3dVector(pos)
    edges.lines = o3d.utility.Vector2iVector(lines)
    edges.colors = o3d.utility.Vector3dVector(colors)

    # node positions
    colors = np.full((len(pos), 3), fill_value=[0.33, 0.33, 0.33])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pos)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # visualise
    _ = o3d.visualization.Visualizer()
    o3d.visualization.draw_geometries([point_cloud, edges])


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
        loc_files = os.listdir(
            os.path.join(project_directory, "preprocessed/featextract/locs")
        )
    except FileNotFoundError:
        raise ValueError("There should be some loc files to open")

    try:
        cluster_files = os.listdir(
            os.path.join(project_directory, "preprocessed/featextract/clusters")
        )
    except FileNotFoundError:
        raise ValueError("There should be some cluster files to open")

    assert loc_files == cluster_files

    # make seaborn plots pretty
    sns.set_style("darkgrid")

    # make output folder
    output_folder = os.path.join(project_directory, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- Analyse cluster features -------
    if not args.neuralnet:
        analyse_manual_feats(project_directory, loc_files, label_map, config, args)
    elif args.neuralnet:
        analyse_nn_feats(project_directory, label_map, config, args)
    else:
        raise ValueError("Should be neural net or manual")


def analyse_manual_feats(
    project_directory,
    loc_files,
    label_map,
    config,
    args,
):
    """Analyse the features of the clusters manually extracted

    Args:
        project_directory (str): Location of the project directory
        loc_files (list): List of the files with the protein
            localisations
        label_map (dict): Map from the label name to number
        config (dict): Configuration for this script
        args (dict): Arguments passed to this script
    """

    # aggregate cluster features into collated df
    dfs = []

    for index, file in enumerate(loc_files):
        # loc_path = os.path.join(
        #   project_directory, f"preprocessed/featextract/locs/{file}"
        # )
        cluster_path = os.path.join(
            project_directory, f"preprocessed/featextract/clusters/{file}"
        )

        cluster_df = pq.read_table(cluster_path)

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
        dfs.append(cluster_df)

    # aggregate dfs into one big df
    df = pl.concat(dfs)
    df = df.to_pandas()

    # get features present in the dataframe
    not_features = ["clusterID", "x_mean", "y_mean", "type", "file_name"]
    features = [x for x in df.columns if x not in not_features]

    # now remove features not selected by user
    user_selected_features = config["features"]
    removed_features = [f for f in features if f not in user_selected_features]
    print("Removed features: ", removed_features)
    features = [f for f in features if f in user_selected_features]
    print("Features analysed: ", features)

    # feature vector
    data_feats = df[features].values

    # label vector
    unique_vals = sorted(df.type.unique())
    labs = sorted(label_map.keys())
    assert labs == unique_vals
    data_labels = df.type.map(label_map).values

    # file names
    names = df.file_name

    # Analyses

    # 1. Plot PCA length/area calculation vs convex hull to compare
    if config["pca_vs_convex_hull"]:
        ax = sns.lineplot(data=df, x="length_pca", y="length_convex_hull")
        ax.set(xlabel="Length (PCA)", ylabel="Length (Convex hull)")
        plt.show()
        # save data to excel sheet
        df_save = df[["length_pca", "length_convex_hull"]]
        df_save_path = os.path.join(
            project_directory, "output/pca_conv_hull_length.csv"
        )
        df_save.to_csv(df_save_path, index=False)
        ax = sns.lineplot(data=df, x="area_pca", y="area_convex_hull")
        plt.show()
        # save data to excel sheet
        df_save = df[["area_pca", "area_convex_hull"]]
        df_save_path = os.path.join(project_directory, "output/pca_conv_hull_area.csv")
        df_save.to_csv(df_save_path, index=False)

    # 2. Save features + cluster/type counts to .csv and plot boxplots of features
    df_save = df[features + ["type", "file_name"]]
    df_save_path = os.path.join(project_directory, "output/cluster_features.csv")
    df_save.to_csv(df_save_path, index=False)

    df_save_pl = pl.from_pandas(df[["type", "file_name"]])
    cluster_counts = df_save_pl["file_name"].value_counts()
    type_counts = df_save_pl["type"].value_counts()
    cluster_counts = df_save_pl.join(cluster_counts, on="file_name")[
        ["file_name", "type", "counts"]
    ].unique()
    df_save_path = os.path.join(project_directory, "output/fov_cluster_count.csv")
    cluster_counts.write_csv(df_save_path)
    df_save_path = os.path.join(project_directory, "output/cluster_type_count.csv")
    type_counts.write_csv(df_save_path)

    # save per fov features grouped by mean with std
    fov_mean = df_save.groupby(["file_name", "type"]).mean()
    fov_std = df_save.groupby(["file_name", "type"]).std()
    fov_output = fov_mean.merge(
        fov_std, on=["file_name", "type"], suffixes=["_mean", "_std"]
    )
    fov_save_path = os.path.join(project_directory, "output/fov_features.csv")
    fov_output.to_csv(fov_save_path, index=True)

    if config["boxplots"]:
        plot_boxplots(features, df)

    X, Y, train_indices_main, val_indices_main, test_indices_main = prep_for_sklearn(
        data_feats, data_labels, names, args
    )

    # Plot UMAP
    if config["umap"]:
        scaler = StandardScaler().fit(X)
        X_umap = scaler.transform(X)
        plot_umap(X_umap, df, config["label_map"])

    # PCA
    if config["pca"]["implement"]:
        scaler = StandardScaler().fit(X)
        X_pca = scaler.transform(X)
        reduced_data = plot_pca(
            X_pca, df, config["label_map"], config["pca"]["n_components"]
        )

    # k-means
    if config["kmeans"]:
        scaler = StandardScaler().fit(X)
        X_kmeans = scaler.transform(X)
        kmeans(X_kmeans, df, config["label_map"])

    # ---------------------------------------------------------------------- #
    # Prediction methods taking in the folds
    # ---------------------------------------------------------------------- #

    # Logistic regression
    if "log_reg" in config.keys():
        parameters = config["log_reg"]
        save_dir = os.path.join(project_directory, "output/log_reg")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_reg(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            features,
            parameters,
            names,
            args,
            None,
            save_dir,
            label_map,
        )

    # Decision tree
    if "dec_tree" in config.keys():
        parameters = config["dec_tree"]
        save_dir = os.path.join(project_directory, "output/dec_tree")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dec_tree(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            features,
            parameters,
            names,
            args,
            None,
            save_dir,
            label_map,
        )

    # K-NN
    if "knn" in config.keys():
        parameters = config["knn"]
        save_dir = os.path.join(project_directory, "output/knn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        knn(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
            None,
            save_dir,
            label_map,
        )

    # SVM
    if "svm" in config.keys():
        parameters = config["svm"]
        save_dir = os.path.join(project_directory, "output/svm")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        svm(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
            None,
            save_dir,
            label_map,
        )


def analyse_nn_feats(project_directory, label_map, config, args):
    """Analyse the features of the clusters from neural network

    Args:
        project_directory (str): Location of the project directory
        label_map (dict): Map from the label name to number
        config (dict): Configuration for this script
        args (dict): Arguments passed to this script

    Raises:
        ValueError: If device specified is neither cpu or gpu OR
            if attention to examine is not correctly specified
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

    model_type = config["model"]

    # only works for locclusternet/locclusternettransformer at the moment
    # 1. To construct datasets we use cluster_net required in model
    # 2. For explainability also assumes uses LocClusterNet
    assert model_type == "locclusternet" or model_type == "locclusternettransformer"

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
    # needs to be from same fold as below
    fold = config["fold"]
    model_name = config["model_name"]
    if not args.automatic:
        model_loc = os.path.join(
            project_directory, "models", f"fold_{fold}", model_name
        )
    elif args.automatic:
        model_dir = os.path.join(project_directory, "models", f"fold_{fold}")
        model_list = os.listdir(model_dir)
        assert len(model_list) == 1
        model_name = model_list[0]
        model_loc = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(model_loc))
    model.to(device)
    model.eval()

    # need to create a homogenous dataset consisting only of clusters from the heterogeneous graph
    data_folder = os.path.join(project_directory, "processed", "featanalysis")

    input_train_folder = os.path.join(
        project_directory, "processed", f"fold_{fold}", "train"
    )
    output_train_folder = os.path.join(data_folder, "train")
    input_val_folder = os.path.join(
        project_directory, "processed", f"fold_{fold}", "val"
    )
    output_val_folder = os.path.join(data_folder, "val")
    input_test_folder = os.path.join(
        project_directory, "processed", f"fold_{fold}", "test"
    )
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

    # get item to evaluate on
    dataitem_idx = config["dataitem"]
    for idx in dataitem_idx:
        cluster_dataitem = cluster_test_set.get(idx)
        loc_dataitem = loc_test_set.get(idx)
        loc_dataitem.to(device)

        # ---- subgraphx -----
        if "subgraphx" in config.keys():
            explainer = SubgraphX(
                cluster_model,
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
            )

            # subgraphx requires output to be logits
            exp = explainer.get_explanation_graph(
                x=cluster_dataitem.x,
                edge_index=cluster_dataitem.edge_index,
                label=cluster_dataitem.y,
                max_nodes=config["subgraphx"]["max_nodes"],
                forward_kwargs={
                    "batch": torch.tensor([0], device=device),
                    "pos": cluster_dataitem.pos,
                    "logits": True,
                },
            )

            # evaluate explanation
            visualise_cluster_explanation(
                cluster_dataitem, exp.node_imp.cpu().numpy(), exp.edge_imp.cpu().numpy()
            )

        # ---- gradcam ----
        # To implement GradCAM we need to remove all MLP layers
        # as gradcam works by getting the weight attribute of the layer
        # but MLP doesn't have this attribute therefore replace all MLP
        # with the parts of it
        # explainer = GradCAM(cluster_model,
        #                     criterion = criterion
        #                     )
        #
        # exp = explainer.get_explanation_graph(
        #    x = cluster_dataitem.x,
        #    edge_index = cluster_dataitem.edge_index,
        #    label=cluster_dataitem.y,
        #    average_variant=True,
        #    forward_kwargs={"batch": torch.tensor([0], device=device)},
        # )
        # input('stop')
        # visualise_cluster_explanation(
        #        cluster_dataitem, exp.node_imp.cpu().numpy(), exp.edge_imp.cpu().numpy()
        #    )

        # ---- guided backprop ----
        if "guided_backprop" in config.keys():
            if config["guided_backprop"]["criterion"] == "nll":
                criterion = torch.nn.functional.nll_loss
            else:
                raise NotImplementedError("This criterion is not implemented")

            explainer = GuidedBP(
                cluster_model,
                criterion,
            )

            # guidedbp doesn't assume logits but we need a criterion between the prediction
            # and label therefore use nll loss and logprobs
            exp = explainer.get_explanation_graph(
                x=cluster_dataitem.x,
                y=cluster_dataitem.y,
                edge_index=cluster_dataitem.edge_index,
                aggregate_node_imp=torch.sum,
                forward_kwargs={
                    "batch": torch.tensor([0], device=device),
                    "pos": cluster_dataitem.pos,
                    "logits": False,
                },
            )

            # scale node importance to between 0 and 1
            min_node = torch.min(exp.node_imp)
            max_node = torch.max(exp.node_imp)
            node_imp = (exp.node_imp - min_node) / (max_node - min_node)
            # set edge importance all to zero as no edge importance from
            # explanation
            edge_index = cluster_dataitem.edge_index.cpu().numpy()
            edge_imp = torch.zeros(edge_index.shape[1])

            visualise_cluster_explanation(
                cluster_dataitem, node_imp.cpu().numpy(), edge_imp.cpu().numpy()
            )

        # ---- pgexplainer ----
        if "pgex" in config.keys():
            max_epochs = config["pgex"]["max_epochs"]
            lr = config["pgex"]["lr"]
            edge_size = config["pgex"]["edge_size"]
            edge_ent = config["pgex"]["edge_ent"]
            temp = config["pgex"]["temp"]
            bias = config["pgex"]["bias"]

            # PGExplainer make it return logprobs
            explainer = Explainer(
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
            explainer.algorithm.mlp.to(device)

            ## Required to first train PGExplainer on the dataset:
            pgex_train_set = torch.utils.data.ConcatDataset(
                [cluster_train_set, cluster_val_set]
            )

            # initialise loader
            train_loader = L.DataLoader(
                pgex_train_set,
                batch_size=1,
                shuffle=True,
                drop_last=True,
            )

            # train pgexplainer
            print("Training")
            for epoch in range(max_epochs):
                total_loss = 0
                items = 0
                for index, item in enumerate(train_loader):
                    loss = explainer.algorithm.train(
                        epoch,
                        cluster_model,
                        item.x,
                        item.edge_index,
                        target=item.y,
                        pos=item.pos,
                        batch=torch.tensor([0], device=device),
                        # return logprobs
                        logits=False,
                    )
                    items += index
                    total_loss += loss
                print(f"Epoch: {epoch}; Loss : {total_loss/items}")

            # explain cluster dataitem
            explanation = explainer(
                cluster_dataitem.x,
                cluster_dataitem.edge_index,
                target=cluster_dataitem.y,
                pos=cluster_dataitem.pos,
                batch=torch.tensor([0], device=device),
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

            # visualise explanation
            visualise_edge_mask(
                cluster_dataitem.pos,
                cluster_dataitem.edge_index,
                explanation.edge_mask,
            )

        # -------- PYTORCH GEO XAI -------------
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
                visualise_edge_mask(
                    cluster_dataitem.pos,
                    cluster_dataitem.edge_index,
                    explanation.edge_mask,
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
                visualise_edge_mask(
                    loc_pos_dict["locs"],
                    loc_edge_index_dict["locs", "in", "clusters"],
                    explanation.edge_mask,
                )

    # ------- BOXPLOT/UMAP/SKLEARN SETUP ---------

    # aggregate cluster features into collated df
    dataset = torch.utils.data.ConcatDataset(
        [cluster_train_set, cluster_val_set, cluster_test_set]
    )

    dfs = []

    # load in gt_label_map
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        gt_label_map = metadata["gt_label_map"]

    gt_label_map = {int(key): val for key, val in gt_label_map.items()}

    for _, data in enumerate(dataset):
        # gt label
        gt_label = int(data.y)
        label = gt_label_map[gt_label]

        # file name
        file_name = data.name + ".parquet"

        # convert to polars
        data = data.x.detach().cpu().numpy()
        cluster_df = pl.DataFrame(data)
        cluster_df = cluster_df.with_columns(pl.lit(label).alias("type"))
        cluster_df = cluster_df.with_columns(pl.lit(f"{file_name}").alias("file_name"))
        dfs.append(cluster_df)

    # aggregate dfs into one big df
    df = pl.concat(dfs)
    df = df.to_pandas()

    # get features present in the dataframe
    not_features = ["type", "file_name"]
    features = [x for x in df.columns if x not in not_features]

    # feature vector
    data_feats = df[features].values

    # label vector
    unique_vals = sorted(df.type.unique())
    labs = sorted(label_map.keys())
    assert labs == unique_vals
    data_labels = df.type.map(label_map).values

    # file names
    names = df.file_name

    # Analyses
    # Plot boxplots of features
    df_save = df[features + ["type"]]
    df_save_path = os.path.join(project_directory, "output/features_nn.csv")
    df_save.to_csv(df_save_path, index=False)
    if config["boxplots"]:
        plot_boxplots(features, df)

    X, Y, train_indices_main, val_indices_main, test_indices_main = prep_for_sklearn(
        data_feats, data_labels, names, args
    )

    # need to ensure no manual features being analysed
    with open("config/process.yaml", "r") as ymlfile:
        process_config = yaml.safe_load(ymlfile)
    cluster_features = process_config["cluster_feat"]
    assert cluster_features is None

    # --------------- UMAP --------------------------
    # Plot UMAP
    if config["umap"]:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        plot_umap(X, df, config["label_map"])

    # ---------------- PCA --------------------------
    # PCA
    if config["pca"]["implement"]:
        scaler = StandardScaler().fit(X)
        X_pca = scaler.transform(X)
        reduced_data = plot_pca(
            X_pca, df, config["label_map"], config["pca"]["n_components"]
        )

    # ---------------- K-MEANS ----------------------
    # k-means
    if config["kmeans"]:
        scaler = StandardScaler().fit(X)
        X_kmeans = scaler.transform(X)
        kmeans(X_kmeans, df, config["label_map"])

    # ------ Prediction methods taking in the folds (sklearn) ----- #
    # train/test indices are list of lists
    # with one list for each fold
    train_indices_main = [train_indices_main[fold]]
    val_indices_main = [val_indices_main[fold]]
    test_indices_main = [test_indices_main[fold]]

    # Logistic regression
    if "log_reg" in config.keys():
        parameters = config["log_reg"]
        save_dir = os.path.join(project_directory, "output/log_reg_nn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_reg(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            features,
            parameters,
            names,
            args,
            fold,
            save_dir,
            label_map,
        )

    # Decision tree
    if "dec_tree" in config.keys():
        parameters = config["dec_tree"]
        save_dir = os.path.join(project_directory, "output/dec_tree_nn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dec_tree(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            features,
            parameters,
            names,
            args,
            fold,
            save_dir,
            label_map,
        )

    # K-NN
    if "knn" in config.keys():
        parameters = config["knn"]
        save_dir = os.path.join(project_directory, "output/knn_nn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        knn(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
            fold,
            save_dir,
            label_map,
        )

    # SVM
    if "svm" in config.keys():
        parameters = config["svm"]
        save_dir = os.path.join(project_directory, "output/svm_nn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        svm(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
            fold,
            save_dir,
            label_map,
        )


def class_report_fn(df, indices):
    """Produce class report for the given dataframe and indices

    Args:
        df (DataFrame): Contains the results
        indices (list): Indices of data to be analysed

    Returns:
        conf_maxtrix (array): Confusion matrix
        f1 (float): F1 score
        acc (float): Accuracy score"""

    # filter dataframe by only test items
    df = df[indices]
    # take mode prediction across all the clusters for each fov
    df = df.to_pandas()
    df = df.groupby("name").agg(lambda x: pd.Series.mode(x)[0])
    df = pl.from_pandas(df)

    # double check that test files agree
    # load config
    # config_path = os.path.join(args.project_directory, "k_fold.yaml")
    # with open(config_path, "r") as ymlfile:
    #    k_fold_config = yaml.safe_load(ymlfile)
    # splits = k_fold_config["splits"]
    # test_fold = splits["test"][fold]
    # assert (sorted(test_fold) == sorted(df_output['name'].to_list()))

    # calculate classification report
    y_true = df["target"].to_list()
    y_pred = df["output"].to_list()
    print(classification_report(y_true, y_pred))

    conf_matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    return conf_matrix, f1, acc


def class_report(predicted, Y, names, train_indices, test_indices, args, fold):
    """Produce report on classification for test set, for particular fold using
    the best model

    Args:
        predicted (array): Predicted data
        Y (array): Target data
        names (list): Names associated with each cluster
        train_indices (list): Indices of the clusters that are part
            of the train set
        test_indices (list): Indices of the clusters that are part
            of the test set
        args (parser args): Arguments passed to the script
        fold (int): Integer representing the fold we are evaluating on

    Returns:
        train_conf_maxtrix (array): Confusion matrix for training set
        test_conf_maxtrix (array): Confusion matrix for test set
        f1_train (float): F1 score for the training set
        acc_train (float): Accuracy score for the training set
        f1_test (float): F1 score for the test set
        acc_test (float): Accuracy score for the test set
    """

    # prediction by the best model
    df_output = pl.DataFrame({"name": names, "output": predicted, "target": Y})

    print(f"--- Classification report (train set) for fold {fold} ---")
    train_confusion_matrix, f1_train, acc_train = class_report_fn(
        df_output, train_indices
    )
    print(f"--- Classification report (test set) for fold {fold} ---")
    test_confusion_matrix, f1_test, acc_test = class_report_fn(df_output, test_indices)

    print("Rows = True; Columns = Prediction")
    return (
        train_confusion_matrix,
        test_confusion_matrix,
        f1_train,
        acc_train,
        f1_test,
        acc_test,
    )


def plot_boxplots(features, df):
    """Plot boxplots of the features

    Args:
        features (list): List of the features in the data to plot
        df (pl.DataFrame): Dataframe with the localisation data"""
    for f in features:
        sns.boxplot(data=df, x=f, y="type")
        plt.show()


def plot_umap(data_feats_scaled, df, label_map):
    """Plot UMAP for the features

    Args:
        data_feats_scaled (array): Features scaled between 0 and 1
        df (pl.DataFrame): Dataframe with the localisation data
        label_map (dict): Keys are real concepts and values are integers
    """

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data_feats_scaled)

    # Plot UMAP - per cluster
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in df.type.map(label_map)],
        label=[x for x in df.type.map(label_map)],
    )
    num_classes = len(label_map.keys())
    patches = [
        mpatches.Patch(color=sns.color_palette()[i], label=list(label_map.keys())[i])
        for i in range(num_classes)
    ]
    plt.legend(handles=patches)
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection of the dataset", fontsize=24)
    plt.show()


def plot_pca(data_feats_scaled, df, label_map, n_components=2):
    """Plot PCA for the features

    Args:
        data_feats_scaled (array): Features scaled between 0 and 1
        df (pl.DataFrame): Dataframe with the localisation data
        label_map (dict): Keys are real concepts and values are integers
        n_components (int): Number of components to retain in PCA

    Returns:
        output_data (array): Output fitted and transformed data
    """

    # transform via PCA
    n_classes = len(label_map.keys())
    reduced_data = PCA(n_components=n_components).fit_transform(data_feats_scaled)
    output_data = reduced_data.copy()

    # convert 2d to 3d if required for plotting
    if reduced_data.shape[1] == 2:
        z = np.ones(reduced_data.shape[0])
        z = np.expand_dims(z, axis=1)
        reduced_data = np.concatenate([reduced_data, z], axis=1)

    # colour clusters according to class
    colors = np.zeros((len(reduced_data), 3))
    for cls in range(n_classes):
        idx = np.argwhere(df.type.map(label_map) == cls)
        colors[idx] = sns.color_palette()[cls]
        print(f"Class {cls} is {sns.color_palette()[cls]}")

    # plot clusters in o3d
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(reduced_data)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # visualise
    _ = o3d.visualization.Visualizer()
    o3d.visualization.draw_geometries([point_cloud])

    return output_data


def kmeans(data_feats_scaled, df, label_map):
    """Plot KMeans for the features

    Args:
        data_feats_scaled (array): Features scaled between 0 and 1
        df (pl.DataFrame): Dataframe with the localisation data
        label_map (dict): Keys are real concepts and values are integers
    """

    n_clusters = len(label_map.keys())
    reduced_data = PCA(n_components=2).fit_transform(data_feats_scaled)
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(reduced_data)

    y_true = df.type.map(label_map).to_numpy()
    y_pred = kmeans.labels_

    print("--- K means report ---")
    print(classification_report(y_true, y_pred))


def prep_for_sklearn(data_feats, data_labels, names, args):
    """Get data ready for sklearn analysis

    Args:
        data_feats (array): Features unscaled
        data_labels (array): Label for each data item
        names (array): Names for each data item
        args (dict): Arguments passed to the script

    Raises:
        ValueError: If overlap between train and test indices

    Returns:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of indices of train data
        val_indices_main (list): List of indices of validation data
        test_indices_main (list): List of indices of test data
    """

    warnings.warn(
        "There must be a config file for k_fold.yaml in the directory for this to work"
    )

    # load config
    config_path = os.path.join(args.project_directory, "config/k_fold.yaml")
    with open(config_path, "r") as ymlfile:
        k_fold_config = yaml.safe_load(ymlfile)

    splits = k_fold_config["splits"]
    train_folds = splits["train"]
    val_folds = splits["val"]
    test_folds = splits["test"]

    df = pl.DataFrame(
        {
            "X": data_feats,
            "Y": data_labels,
            "name": names,
        }
    )

    # get indices of train/test for CV
    train_indices_main = []
    val_indices_main = []
    test_indices_main = []

    for index, train_fold in enumerate(train_folds):
        val_fold = val_folds[index]
        test_fold = test_folds[index]

        train_bool = df["name"].is_in(train_fold).to_list()
        val_bool = df["name"].is_in(val_fold).to_list()
        test_bool = df["name"].is_in(test_fold).to_list()

        train_indices = np.where(train_bool)[0]
        val_indices = np.where(val_bool)[0]
        test_indices = np.where(test_bool)[0]

        train_indices_main.append(train_indices)
        val_indices_main.append(val_indices)
        test_indices_main.append(test_indices)

        if any(i in train_indices for i in test_indices):
            raise ValueError("Should not share common values")
        if any(i in train_indices for i in val_indices):
            raise ValueError("Should not share common values")
        if any(i in test_indices for i in val_indices):
            raise ValueError("Should not share common values")

    num_features = len(df["X"][0])
    print("Num features: ", num_features)
    warnings.warn(
        "Be careful, if analysing neural net features"
        "Is this the number of features you expect"
        "Did this task use manual features as well"
    )

    X = df["X"].to_list()
    Y = df["Y"].to_list()

    return X, Y, train_indices_main, val_indices_main, test_indices_main


def fold_results(
    X,
    Y,
    model,
    train_indices_main,
    test_indices_main,
    names,
    args,
    fold,
    save_dir,
    label_map,
):
    """foo

    Args:
        X (array): Input data
        Y (array): Target data
        model (sklearn model): Model to be evaluated
        train_indices_main (list): List of the indices of the training data
        test_indices_main (list): List of the indices of the test data
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): denotes the fold we are evaluating or is None
        save_dir (string): directory to save results to
        label_map (dict): from real name to integer

    Raises:
        ValueError: If I (Oli) have made a mistake
    """

    print("---- Fit to the specified fold or each fold ----")
    # set up arrays
    X = np.array(X)
    Y = np.array(Y)
    cv = iter(zip(train_indices_main, test_indices_main))

    if fold is not None:
        assert type(fold) is int
        fold_index = fold
        evaluated = False
    else:
        fold_index = 0

    train_results = {"fold": [], "f1": [], "acc": []}
    test_results = {"fold": [], "f1": [], "acc": []}

    for train_fold, test_fold in cv:
        train_fold = np.array(train_fold)
        test_fold = np.array(test_fold)

        # scale data
        scaler = StandardScaler().fit(X[train_fold])
        X = scaler.transform(X)

        model = model.fit(X[train_fold], Y[train_fold])
        output = model.predict(X)
        (
            train_report,
            test_report,
            f1_train,
            acc_train,
            f1_test,
            acc_test,
        ) = class_report(
            output,
            Y,
            names,
            train_fold,
            test_fold,
            args,
            fold_index,
        )
        col_names = list(dict(sorted(label_map.items())).keys())

        # append results
        train_results["f1"].append(f1_train)
        train_results["acc"].append(acc_train)
        test_results["f1"].append(f1_test)
        test_results["acc"].append(acc_test)
        train_results["fold"].append(fold_index)
        test_results["fold"].append(fold_index)

        # save train results
        df_save = pd.DataFrame(train_report, columns=col_names, index=col_names)
        df_save_path = os.path.join(save_dir, f"{fold_index}_fov_train.csv")
        df_save.to_csv(df_save_path)

        # save test results
        df_save = pd.DataFrame(test_report, columns=col_names, index=col_names)
        df_save_path = os.path.join(save_dir, f"{fold_index}_fov_test.csv")
        df_save.to_csv(df_save_path)

        # if fold is specified should only enter iterator once
        if fold is not None:
            if not evaluated:
                evaluated = True
            else:
                raise ValueError("Error from designer")

        fold_index += 1

    # save overall
    df_save = pd.DataFrame(train_results)
    df_save_path = os.path.join(save_dir, f"fov_train.csv")
    df_save.to_csv(df_save_path, index=False)

    # save overall
    df_save = pd.DataFrame(test_results)
    df_save_path = os.path.join(save_dir, f"fov_test.csv")
    df_save.to_csv(df_save_path, index=False)


def log_reg(
    X,
    Y,
    train_indices_main,
    val_indices_main,
    test_indices_main,
    features,
    parameters,
    names,
    args,
    fold,
    save_dir,
    label_map,
):
    """Perform logistic reggression on the dataset

    Args:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of the indices of the training data
        val_indices_main (list): List of the indices of the validation data
        test_indices_main (list): List of the indices of the test data
        features (list): List of features analysing
        parameters (dict): Parameters to try logistic regression for
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): If specified denotes the fold we are evaluating
        save_dir (str): Folder to save results to
        label_map (dict): From real names to integers

    Raises:
        ValueError: If training and test sets overlap

    Returns:
        best_model (estimator): The model which gave the highest score

    """
    cv = iter(zip(train_indices_main, val_indices_main))

    model = LogisticRegression(max_iter=1000)
    clf = GridSearchCV(model, parameters, cv=cv)

    print("-----Log reg.-------")
    print("--------------------")

    clf.fit(X, Y)
    df = pd.DataFrame(clf.cv_results_)
    df = df[
        [
            "param_C",
            "param_penalty",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ]
    print("------ Best parameters (ignore values) (results are on validation set) ---")
    df = df.sort_values(by=["rank_test_score"])
    print(df)
    save_df_path = os.path.join(save_dir, "best_params.csv")
    df.to_csv(save_df_path, index=False)

    best_model = clf.best_estimator_
    best_feats = dict(zip(features, best_model.coef_[0].tolist()))
    print("------ Coefficients --------")
    coeffs = sorted(best_feats.items(), key=lambda x: abs(x[1]), reverse=True)
    print(coeffs)
    coeff_df = pd.DataFrame(coeffs)
    save_df_path = os.path.join(save_dir, "best_coeffs.csv")
    coeff_df.to_csv(save_df_path, index=False)

    model = LogisticRegression(max_iter=1000, **clf.best_params_)

    train_indices = train_indices_main.copy()
    val_indices = val_indices_main.copy()
    test_indices = test_indices_main.copy()

    for index, value in enumerate(train_indices):
        train_indices[index] = np.append(value, val_indices[index])
        if any(i in train_indices[index] for i in test_indices[index]):
            raise ValueError("Should not share common values")

    fold_results(
        X, Y, model, train_indices, test_indices, names, args, fold, save_dir, label_map
    )

    return best_model


def dec_tree(
    X,
    Y,
    train_indices_main,
    val_indices_main,
    test_indices_main,
    features,
    parameters,
    names,
    args,
    fold,
    save_dir,
    label_map,
):
    """Perform decision tree on the dataset

    Args:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of the indices of the training data
        val_indices_main (list): List of the indices of the validation data
        test_indices_main (list): List of the indices of the test data
        features (list): List of features analysing
        parameters (dict): Parameters to try decision tree for
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): If specified denotes the fold we are evaluating
        save_dir (str): Folder to save results to
        label_map (dict): From real names to integers

    Raises:
        ValueError: If training and test sets overlap

    Returns:
        best_model (estimator): The model which gave the highest score

    """

    cv = iter(zip(train_indices_main, val_indices_main))

    model = DecisionTreeClassifier()

    clf = GridSearchCV(model, parameters, cv=cv)

    print("-----Dec tree.------")
    print("--------------------")

    clf.fit(X, Y)
    df = pd.DataFrame(clf.cv_results_)
    df = df[
        [
            "param_max_depth",
            "param_max_features",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ]
    print("------ Best parameters (ignore values) (results are on validation set) ---")
    df = df.sort_values(by=["rank_test_score"])
    print(df)
    save_df_path = os.path.join(save_dir, "best_params.csv")
    df.to_csv(save_df_path, index=False)

    best_model = clf.best_estimator_
    best_feats = dict(zip(features, best_model.feature_importances_.tolist()))
    print("------ Coefficients --------")
    coeffs = sorted(best_feats.items(), key=lambda x: abs(x[1]), reverse=True)
    print(coeffs)
    coeff_df = pd.DataFrame(coeffs)
    save_df_path = os.path.join(save_dir, "best_coeffs.csv")
    coeff_df.to_csv(save_df_path, index=False)

    model = DecisionTreeClassifier(**clf.best_params_)

    train_indices = train_indices_main.copy()
    val_indices = val_indices_main.copy()
    test_indices = test_indices_main.copy()

    for index, value in enumerate(train_indices):
        train_indices[index] = np.append(value, val_indices[index])
        if any(i in train_indices[index] for i in test_indices[index]):
            raise ValueError("Should not share common values")

    fold_results(
        X, Y, model, train_indices, test_indices, names, args, fold, save_dir, label_map
    )

    return best_model


def svm(
    X,
    Y,
    train_indices_main,
    val_indices_main,
    test_indices_main,
    parameters,
    names,
    args,
    fold,
    save_dir,
    label_map,
):
    """Perform svm on the dataset

    Args:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of the indices of the training data
        val_indices_main (list): List of the indices of the validation data
        test_indices_main (list): List of the indices of the test data
        parameters (dict): Parameters to try svm for
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): If specified denotes the fold we are evaluating
        save_dir (str): Folder to save results to
        label_map (dict): From real names to integers

    Raises:
        ValueError: If training and test sets overlap

    Returns:
        best_model (estimator): The model which gave the highest score

    """

    cv = iter(zip(train_indices_main, val_indices_main))

    model = SVC()

    clf = GridSearchCV(model, parameters, cv=cv, verbose=4)

    print("--------SVM---------")
    print("--------------------")

    clf.fit(X, Y)
    df = pd.DataFrame(clf.cv_results_)
    df = df[
        [
            "param_C",
            "param_kernel",
            "param_gamma",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ]
    print("------ Best parameters (ignore values) (results are on validation set) ---")
    df = df.sort_values(by=["rank_test_score"])
    print(df)
    save_df_path = os.path.join(save_dir, "best_params.csv")
    df.to_csv(save_df_path, index=False)

    best_model = clf.best_estimator_

    model = SVC(**clf.best_params_)

    train_indices = train_indices_main.copy()
    val_indices = val_indices_main.copy()
    test_indices = test_indices_main.copy()

    for index, value in enumerate(train_indices):
        train_indices[index] = np.append(value, val_indices[index])
        if any(i in train_indices[index] for i in test_indices[index]):
            raise ValueError("Should not share common values")

    fold_results(
        X, Y, model, train_indices, test_indices, names, args, fold, save_dir, label_map
    )

    return best_model


def knn(
    X,
    Y,
    train_indices_main,
    val_indices_main,
    test_indices_main,
    parameters,
    names,
    args,
    fold,
    save_dir,
    label_map,
):
    """Perform knn on the dataset

    Args:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of the indices of the training data
        val_indices_main (list): List of the indices of the validation data
        test_indices_main (list): List of the indices of the test data
        parameters (dict): Parameters to try knn for
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): If specified denotes the fold we are evaluating
        save_dir (str): Folder to save results to
        label_map (dict): From real names to integers

    Raises:
        ValueError: If training and test sets overlap

    Returns:
        best_model (estimator): The model which gave the highest score

    """

    cv = iter(zip(train_indices_main, val_indices_main))

    model = KNeighborsClassifier()

    clf = GridSearchCV(model, parameters, cv=cv)

    print("--------KNN---------")
    print("--------------------")

    clf.fit(X, Y)
    df = pd.DataFrame(clf.cv_results_)
    df = df[
        [
            "param_n_neighbors",
            # "param_weights",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ]
    print("------ Best parameters (ignore values) (results are on validation set) ---")
    df = df.sort_values(by=["rank_test_score"])
    print(df)
    save_df_path = os.path.join(save_dir, "best_params.csv")
    df.to_csv(save_df_path, index=False)

    best_model = clf.best_estimator_

    model = KNeighborsClassifier(**clf.best_params_)

    train_indices = train_indices_main.copy()
    val_indices = val_indices_main.copy()
    test_indices = test_indices_main.copy()

    for index, value in enumerate(train_indices):
        train_indices[index] = np.append(value, val_indices[index])
        if any(i in train_indices[index] for i in test_indices[index]):
            raise ValueError("Should not share common values")

    fold_results(
        X, Y, model, train_indices, test_indices, names, args, fold, save_dir, label_map
    )

    return best_model


# save yaml file
# yaml_save_loc = os.path.join(project_directory, "featextract.yaml")
# with open(yaml_save_loc, "w") as outfile:
#    yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
