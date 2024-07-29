"""Feature analysis module

Module takes in the .parquet files and analyses features

Config file at top specifies the analyses we want to run
"""

import argparse
from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.evaluation import XCollector
import json
import os
import time
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import open3d as o3d
from locpix_points.data_loading import datastruc
from locpix_points.models.cluster_nets import ClusterNetHomogeneous
from locpix_points.models.cluster_nets import parse_data
from locpix_points.models import model_choice
from locpix_points.scripts.visualise import visualise_torch_geometric
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch_geometric.loader as L
from torch_geometric.explain import (
    Explainer,
    PGExplainer,
    AttentionExplainer,
    metric,
    CaptumExplainer,
)
from torch_geometric.utils import to_undirected, contains_self_loops
from torch_geometric.utils.convert import to_networkx, from_networkx
import umap
import warnings
import yaml


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

    warnings.warn(
        "Assumes\
                      1. Using a ClusterNet \
                      2. Embedding is in a final cluster encoder layer\
                      3. Graph level explanation"
    )

    # need to create a model that acts on the homogeneous data for cluster and locs
    cluster_model = ClusterNetHomogeneous(model.cluster_net, config[model_type])
    torch.save(
        cluster_model, os.path.join(project_directory, f"output/cluster_model.pt")
    )
    cluster_model.to(device)
    cluster_model.eval()

    # ------- GRAPHXAI --------
    # train pgexplainer
    if "pgex" in config.keys():
        pg_explainer = train_pgex(
            config, cluster_model, cluster_train_set, cluster_val_set, device
        )
        torch.save(
            pg_explainer, os.path.join(project_directory, f"output/pg_explainer.pt")
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


def train_pgex(config, cluster_model, cluster_train_set, cluster_val_set, device):
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

    return pg_explainer


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
        data.to(device)

        # file name
        file_name = data.name + ".parquet"

        # forward through network
        logits = cluster_model(
            data.x,
            data.edge_index,
            torch.tensor([0], device=device),
            data.pos,
            logits=True,
        )

        prediction = logits.argmax(-1).item()
        labels.append(gt_label)
        predictions.append(prediction)

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


def generate_umap_embedding(X, min_dist, n_neighbours):
    """Run UMAP

    Args:
        X (array): Array to fit to
        min_dist (float): Distance for umap
        n_neighbours (int): n-neighbours for umap

    Returns:
        embedding (array): UMAP embedding"""

    reducer = umap.UMAP(
        min_dist=min_dist,
        n_neighbors=n_neighbours,
    )
    embedding = reducer.fit_transform(X)

    return embedding


def visualise_umap_embedding(embedding, df, label_map):
    """Visualise UMAP results

    Args:
        embedding (array): UMAP embedding
        df (dataframe): Dataframe with data in
        label_map (dict): Map from numbers to concepts"""

    # Plot UMAP - per cluster
    plt.close("all")
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in df.type.map(label_map)],
        label=[x for x in df.type.map(label_map)],
        s=5,
    )
    num_classes = len(label_map.keys())
    patches = [
        mpatches.Patch(color=sns.color_palette()[i], label=list(label_map.keys())[i])
        for i in range(num_classes)
    ]
    cursor = mplcursors.cursor(hover=False)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(f"{df.file_name[sel.index]}")
    )
    plt.legend(handles=patches)
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection of the dataset", fontsize=24)
    plt.show()


def generate_pca_embedding(X, n_components):
    """Run PCA

    Args:
        X (array): Array to fit to
        n_components (int): n-components for pca

    Returns:
        reduced_data (array): PCA transformed data"""

    # transform via PCA
    reduced_data = PCA(n_components=n_components).fit_transform(X)

    return reduced_data


def visualise_pca_embedding(pca_embedding, df, label_map):
    """Visualise PCA embedding

    Args:
        pca_embedding (array): PCA embedded data
        df (dataframe): Dataframe with data in
        label_map (dict): Map from numbers to concepts"""

    n_classes = len(label_map.keys())

    # convert 2d to 3d if required for plotting
    if pca_embedding.shape[1] == 2:
        z = np.ones(pca_embedding.shape[0])
        z = np.expand_dims(z, axis=1)
        pca_embedding = np.concatenate([pca_embedding, z], axis=1)

    # colour clusters according to class
    colors = np.zeros((len(pca_embedding), 3))
    for cls in range(n_classes):
        idx = np.argwhere(df.type.map(label_map) == cls)
        colors[idx] = sns.color_palette()[cls]
        class_label = list(label_map.keys())[list(label_map.values()).index(cls)]
        print(
            f"Class {class_label} is RGB colour: {sns.color_palette()[cls]}", flush=True
        )

    # plot clusters in o3d
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pca_embedding)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # visualise
    _ = o3d.visualization.Visualizer()
    o3d.visualization.draw_geometries([point_cloud])


def k_means_fn(X, df, label_map):
    """Run KMeans

    Args:
        X (array): Array to fit to
        df (dataframe): Dataframe with data in
        label_map (dict): Map from numbers to concepts"""

    n_clusters = len(label_map.keys())
    y_true = df.type.map(label_map).to_numpy()

    # with PCA reduction
    reduced_data = PCA(n_components=2).fit_transform(X)
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters)
    kmeans.fit(reduced_data)
    y_pred = kmeans.labels_

    print("--- K means report (with PCA reduction to 2D) ---")
    print(classification_report(y_true, y_pred))

    # without PCA reduction
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters)
    kmeans.fit(X)
    y_pred = kmeans.labels_

    print("--- K means report (NO PCA reduction) ---")
    print(classification_report(y_true, y_pred))


def get_prediction(
    file_name,
    cluster_model,
    cluster_train_set,
    loc_train_set,
    cluster_val_set,
    loc_val_set,
    cluster_test_set,
    loc_test_set,
    project_directory,
    device,
    gt_label_map,
):
    """Get the prediction for a file using the cluster model

    Args:
        file_name (string): Name of the file
        cluster_model (pyg model): PyGeometric model
        cluster_train_set (pyg dataset): Training set with clusters having
            gone through a locnet
        loc_train_set (pyg dataset): Training set with raw localisations
        cluster_val_set (pyg dataset): Validation set with clusters having
            gone through a locnet
        loc_val_set (pyg dataset): Validation set with raw localisations
        cluster_test_set (pyg dataset): Test set with clusters having
            gone through a locnet
        loc_test_set (pyg dataset): Test set with raw localisations
        project_directory (string): Location of the project directory
        device (string): Device to run on
        gt_label_map (dict): Map from labels to real concepts

    Returns:
        cluster_dataitem (pyg dataitem): Cluster graph embedded into each node
        loc_dataitem (pyg dataitem): Cluster graph raw
        prediction (float): Predicted label
    """

    # load in gt_label_map
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        gt_label_map = metadata["gt_label_map"]

    gt_label_map = {int(key): val for key, val in gt_label_map.items()}

    train_file_map_path = os.path.join(
        project_directory, f"processed/featanalysis/train/file_map.csv"
    )
    val_file_map_path = os.path.join(
        project_directory, f"processed/featanalysis/val/file_map.csv"
    )
    test_file_map_path = os.path.join(
        project_directory, f"processed/featanalysis/test/file_map.csv"
    )

    train_file_map = pd.read_csv(train_file_map_path)
    val_file_map = pd.read_csv(val_file_map_path)
    test_file_map = pd.read_csv(test_file_map_path)

    train_out = train_file_map[train_file_map["file_name"] == file_name]
    val_out = val_file_map[val_file_map["file_name"] == file_name]
    test_out = test_file_map[test_file_map["file_name"] == file_name]

    if len(train_out) > 0:
        file_name = train_out["idx"].values[0]
        cluster_dataitem = cluster_train_set.get(file_name)
        loc_dataitem = loc_train_set.get(file_name)

    if len(val_out) > 0:
        file_name = val_out["idx"].values[0]
        cluster_dataitem = cluster_val_set.get(file_name)
        loc_dataitem = loc_val_set.get(file_name)

    if len(test_out) > 0:
        file_name = test_out["idx"].values[0]
        cluster_dataitem = cluster_test_set.get(file_name)
        loc_dataitem = loc_test_set.get(file_name)

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
    # print(f"Item {idx}")
    print("Predicted label: ", gt_label_map[prediction])
    print("GT label: ", gt_label_map[cluster_dataitem.y.cpu().item()])

    return cluster_dataitem, loc_dataitem, prediction


def subgraph_eval(cluster_model, device, config, cluster_dataitem, prediction):
    """Evaluate SubgraphX explainability algo

    Args:
        cluster_model (PyG model): PyG model acts on the clusters
        device (string): Device to run on
        config (dict): Parameters for algo
        cluster_dataitem (pyg dataitem): Cluster graph to pass through network
        prediction (float): Prediction for the cluster graph

    Returns:
        subgraph (PyG graph): The induced subgraph from the important structure
        complement (PyG graph): The complement to the subgraph"""

    print("Subgraphx...")
    explainer = SubgraphX(
        cluster_model,
        num_classes=config["num_classes"],
        device=device,
        explain_graph=True,
        rollout=config["rollout"],
        min_atoms=config["min_atoms"],
        c_puct=config["c_puct"],
        expand_atoms=config["expand_atoms"],
        high2low=config["high2low"],
        local_radius=config["local_radius"],
        sample_num=config["sample_num"],
        reward_method=config["reward_method"],
        subgraph_building_method=config["subgraph_building_method"],
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
        max_nodes=config["max_nodes"],
    )

    # process explanation results
    explanation_results = explanation_results[prediction]
    explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)
    tree_node_x = find_closest_node_result(
        explanation_results, max_nodes=config["max_nodes"]
    )

    # generate metrics for explanation
    nodelist = tree_node_x.coalition
    node_imp = torch.zeros(len(cluster_dataitem.pos))
    node_imp[nodelist] = 1.0
    x_collector = XCollector()
    x_collector.collect_data(tree_node_x.coalition, related_preds, label=prediction)

    # print metrics for explanation
    # print(f"Positive fidelity closer to 1 better: {x_collector.fidelity:.4f})")
    # print(f"Negative fidelity closer to 0 better: {x_collector.fidelity_inv:.4f})")
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

    # alternative fidelity measures
    subgraph, complement = custom_fidelity_measure(
        cluster_model, cluster_dataitem, node_imp, "node", device
    )

    return subgraph, complement


def pgex_eval(cluster_model, pg_explainer, cluster_dataitem, device, config):
    """Evaluate PGExplainer explainability algo

    Args:
        cluster_model (PyG model): PyG model acts on the clusters
        pg_explainer (PGExplainer model): PyG explainer trained to explain predictions
        device (string): Device to run on
        config (dict): Parameters for algo
        cluster_dataitem (pyg dataitem): Cluster graph to pass through network"""

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
    # pos_fid, neg_fid = metric.fidelity(pg_explainer, explanation)
    # print(f"Positive fidelity closer to 1 better: {pos_fid})")
    # print(f"Negative fidelity closer to 0 better: {neg_fid})")
    unf = metric.unfaithfulness(pg_explainer, explanation)
    print(f"Unfaithfulness, closer to 0 better {unf}")

    # visualise explanation
    visualise_explanation(
        cluster_dataitem.pos,
        cluster_dataitem.edge_index,
        node_imp=None,
        edge_imp=explanation.edge_mask.to(device),
    )

    # alternative fidelity measure
    subgraph, complement = custom_fidelity_measure(
        cluster_model, cluster_dataitem, explanation.edge_mask, "edge", device
    )


def attention_eval(cluster_model, config, cluster_dataitem, device, loc_dataitem):
    """Evaluate attention explainability algo

    Args:
        cluster_model (PyG model): PyG model acts on the clusters
        device (string): Device to run on
        config (dict): Parameters for algo
        cluster_dataitem (pyg dataitem): Cluster graph to pass through network
        loc_dataitem (pyg dataitem): Localisation graph to pass through network

    Raises:
        NotImplementedError: If try to run attention on Loc or
            LocCluster instead of cluster


    Returns:
        subgraph (PyG graph): The induced subgraph from the important structure
        complement (PyG graph): The complement to the subgraph"""

    scale = config["scale"]
    reduce = config["reduce"]

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
        # pos_fid, neg_fid = metric.fidelity(explainer, explanation)
        # print(f"Positive fidelity closer to 1 better: {pos_fid})")
        # print(f"Negative fidelity closer to 0 better: {neg_fid})")
        unf = metric.unfaithfulness(explainer, explanation)
        print(f"Unfaithfulness, closer to 0 better {unf}")
        visualise_explanation(
            cluster_dataitem.pos,
            cluster_dataitem.edge_index,
            node_imp=None,
            edge_imp=explanation.edge_mask.to(device),
        )

        # this commented out code removes important edges that are self loops
        # loop_mask = cluster_dataitem.edge_index[0] == cluster_dataitem.edge_index[1]
        # explanation.edge_mask = torch.where(~loop_mask, explanation.edge_mask, 0.0)

        # alternative fidelity measure
        subgraph, complement = custom_fidelity_measure(
            cluster_model, cluster_dataitem, explanation.edge_mask, "edge", device
        )

        return subgraph, complement

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
        # pos_fid, neg_fid = metric.fidelity(explainer, explanation)
        # print(f"Positive fidelity closer to 1 better: {pos_fid})")
        # print(f"Negative fidelity closer to 0 better: {neg_fid})")
        unf = metric.unfaithfulness(explainer, explanation)
        print(f"Unfaithfulness, closer to 0 better {unf}")
        visualise_explanation(
            loc_pos_dict["locs"],
            loc_edge_index_dict["locs", "in", "clusters"],
            node_imp=None,
            edge_imp=explanation.edge_mask.to(device),
        )


def saliency_eval(cluster_model, config, cluster_dataitem, device):
    """Evaluate saliency explainability algo

    Args:
        cluster_model (PyG model): PyG model acts on the clusters
        device (string): Device to run on
        config (dict): Parameters for algo
        cluster_dataitem (pyg dataitem): Cluster graph to pass through network

    Returns:
        subgraph (PyG graph): The induced subgraph from the important structure
        complement (PyG graph): The complement to the subgraph"""

    explainer = Explainer(
        model=cluster_model,
        algorithm=CaptumExplainer(attribution_method="Saliency"),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            # return logprobs
            return_type="log_probs",
        ),
    )

    explanation = explainer(
        x=cluster_dataitem.x,
        edge_index=cluster_dataitem.edge_index,
        target=None,  # attention doesn't use the target nor the return type which is used to generate the target therefore this argument is irrelevant
        # kwargs passed to model
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
    # pos_fid, neg_fid = metric.fidelity(explainer, explanation)
    # print(f"Positive fidelity closer to 1 better: {pos_fid})")
    # print(f"Negative fidelity closer to 0 better: {neg_fid})")
    unf = metric.unfaithfulness(explainer, explanation)
    print(f"Unfaithfulness, closer to 0 better {unf}")
    visualise_explanation(
        cluster_dataitem.pos,
        cluster_dataitem.edge_index,
        node_imp=None,
        edge_imp=explanation.edge_mask.to(device),
    )

    # this commented out code removes important edges that are self loops
    # loop_mask = cluster_dataitem.edge_index[0] == cluster_dataitem.edge_index[1]
    # explanation.edge_mask = torch.where(~loop_mask, explanation.edge_mask, 0.0)

    # alternative fidelity measure
    subgraph, complement = custom_fidelity_measure(
        cluster_model, cluster_dataitem, explanation.edge_mask, "edge", device
    )

    return subgraph, complement


def custom_fidelity_measure(
    cluster_model, cluster_dataitem, imp_list, node_or_edge, device
):
    graph_pred = cluster_model(
        cluster_dataitem.x,
        cluster_dataitem.edge_index,
        torch.tensor([0], device=device),
        cluster_dataitem.pos,
        logits=False,
    )
    graph_pred = torch.exp(graph_pred)
    # probability of the predicted class
    prediction = graph_pred.max(-1)
    predicted_index = prediction.indices
    whole_graph_prob = prediction.values

    imp_list = imp_list.to(device)

    subgraph, complement = induced_subgraph(
        cluster_dataitem, imp_list, node_or_edge=node_or_edge
    )

    subgraph.to(device)
    complement.to(device)

    complement_pred = cluster_model(
        complement.x,
        complement.edge_index,
        torch.tensor([0], device=device),
        complement.pos,
        logits=False,
    )
    complement_pred = torch.exp(complement_pred)
    complement_pred = complement_pred.squeeze()
    complement_prob = complement_pred[predicted_index]

    subgraph_pred = cluster_model(
        subgraph.x,
        subgraph.edge_index,
        torch.tensor([0], device=device),
        subgraph.pos,
        logits=False,
    )
    subgraph_pred = torch.exp(subgraph_pred)
    subgraph_pred = subgraph_pred.squeeze()
    subgraph_prob = subgraph_pred[predicted_index]

    pos_fid = abs(whole_graph_prob - complement_prob)
    neg_fid = abs(whole_graph_prob - subgraph_prob)

    print("Positive fidelity", pos_fid)
    print("Negative fidelity", neg_fid)

    return subgraph, complement


def gradcam_eval(cluster_model, cluster_dataitem, config, device):
    """Evaluate GradCam explainability algo

    Args:
        cluster_model (PyG model): PyG model acts on the clusters
        device (string): Device to run on
        config (dict): Parameters for algo
        cluster_dataitem (pyg dataitem): Cluster graph to pass through network"""

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
    x_collector.collect_data(hard_edge_masks, related_preds, label=cluster_dataitem.y)

    node_imp = None

    # print metrics for explanation
    print(f"Positive fidelity closer to 1 better: {x_collector.fidelity:.4f})")
    print(f"Negative fidelity closer to 0 better: {x_collector.fidelity_inv:.4f})")
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

    # are self loops present
    print("Contains self loops: ", contains_self_loops(edge_index))

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
        if ind_zero.ndim != 0:
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


def induced_subgraph(data, imp_list, node_or_edge="node"):
    """Return the induced subgraph and its complement

    Args:
        data (pyg graph): Graph we want to induce
        imp_list (tensor): Tensor of 0s and 1s corresponding to
            unimportant and important nodes/edges
        node_or_edge (string): If node we induce node subgraph
            and if edge we induce edge subgraph

    Returns:
        subgraph (PyG graph): The induced subgraph from the important structure
        complement (PyG graph): The complement to the subgraph

    Raises:
        ValueError: If induced subgraph is whole graph OR no important nodes/edges"""

    if sum(imp_list) == 0:
        raise ValueError("No important edges/nodes")

    imp_list = imp_list.bool()
    non_imp_list = torch.where(imp_list == False, True, False)

    if node_or_edge == "node":
        # automatically relabelled
        subgraph = data.subgraph(imp_list)
        if subgraph.num_nodes == data.num_nodes:
            raise ValueError(
                "No complement graph as induced subgraph is the whole graph"
            )
        else:
            complement_graph = data.subgraph(non_imp_list)

        return subgraph, complement_graph

    elif node_or_edge == "edge":
        nx_graph = to_networkx(data, node_attrs=["x", "pos"])
        include_edges = data.edge_index.T[imp_list].cpu().numpy()
        include_edges = list(map(tuple, include_edges))
        non_include_edges = data.edge_index.T[non_imp_list].cpu().numpy()
        non_include_edges = list(map(tuple, non_include_edges))
        subgraph = nx_graph.edge_subgraph(include_edges)
        subgraph_pyg = from_networkx(subgraph, group_node_attrs=["x", "pos"])
        x = subgraph_pyg.x[:, :-2]
        pos = subgraph_pyg.x[:, -2:]
        subgraph_pyg.x = x
        subgraph_pyg.pos = pos
        if subgraph.nodes == nx_graph.nodes:
            raise ValueError(
                "No complement graph as induced subgraph is the whole graph"
            )
        else:
            warnings.warning(
                "As the graphs are directed - it may still appear that the important edge is in the complement BUT this will be the edge in the other direction i.e. if two edges between two nodes and only one is important, visualising the graph and complement will appear the same between these nodes"
            )
            complement_graph = nx_graph.edge_subgraph(non_include_edges)
            complement_graph_pyg = from_networkx(
                complement_graph, group_node_attrs=["x", "pos"]
            )
            x = complement_graph_pyg.x[:, :-2]
            pos = complement_graph_pyg.x[:, -2:]
            complement_graph_pyg.x = x
            complement_graph_pyg.pos = pos

            # complement induced by the nodes not in the subgraph below
            # e = list(subgraph.nodes)
            # nx_graph.remove_nodes_from(e)
            # complement_graph_pyg = from_networkx(
            #    nx_graph, group_node_attrs=["x", "pos"]
            # )
            # x = complement_graph_pyg.x[:, :-2]
            # pos = complement_graph_pyg.x[:, -2:]
            # complement_graph_pyg.x = x
            # complement_graph_pyg.pos = pos

        return subgraph_pyg, complement_graph_pyg

    else:
        raise ValueError("node or edge should be node or edge")


if __name__ == "__main__":
    main()
