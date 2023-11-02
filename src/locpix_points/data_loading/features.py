""" Features module

Extracts features for the data
"""

import torch
import polars as pl
import numpy as np
import itertools
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import warnings


def load_loc_cluster(
    data,
    loc_table,
    cluster_table,
    loc_feat,
    cluster_feat,
    min_feat_locs,
    max_feat_locs,
    min_feat_clusters,
    max_feat_clusters,
    kneighbours,
):
    """Load in position, features and edge index to each node

    Args:
        data (torch_geometric data) : Data item to
            load position, features and edge index into
        loc_table (parquet arrow table) : Localisation data
            in form of parquet file
        cluster_table (parquet arrow table) : Cluster data
            in form of parquet file
        loc_feat (list) : List of features to consider in localisation dataset
        cluster_feat (list) : List of features to consider in cluster dataset
        min_feat_locs (dict) : Minimum values of features for the locs training dataset
        max_feat_locs (dict) : Maxmimum values of features over locs training dataset
        min_feat_clusters (dict) : Minimum values of features for the clusters training dataset
        max_feat_clusters (dict) : Maxmimum values of features over clusters training dataset
        kneighbours (int) : How many nearest neighbours to consider constructing knn graph for
            cluster dataset
    Returns:
        data (torch_geometric data) : Data with
            position and feature for eacch node"""

    # load in positions
    x_locs = torch.tensor(loc_table["x"].to_numpy())
    y_locs = torch.tensor(loc_table["y"].to_numpy())
    loc_coords = torch.stack((x_locs, y_locs), dim=1)
    data["locs"].pos = loc_coords.float()

    x_clusters = torch.tensor(cluster_table["x_mean"].to_numpy())
    y_clusters = torch.tensor(cluster_table["y_mean"].to_numpy())
    cluster_coords = torch.stack((x_clusters, y_clusters), dim=1)
    data["clusters"].pos = cluster_coords.float()

    # load in features
    assert list(min_feat_locs.keys()) == loc_feat
    assert list(max_feat_locs.keys()) == loc_feat
    assert list(min_feat_clusters.keys()) == cluster_feat
    assert list(max_feat_clusters.keys()) == cluster_feat

    feat_data = torch.tensor(loc_table.select(loc_feat).to_pandas().to_numpy())
    min_vals = torch.tensor(list(min_feat_locs.values()))
    max_vals = torch.tensor(list(max_feat_locs.values()))
    feat_data = (feat_data - min_vals) / (max_vals - min_vals)
    # clamp needed if val/test data has min/max greater than train set min/max
    feat_data = torch.clamp(feat_data, min=0, max=1)
    data["locs"].x = feat_data.float()  # might not need .float()

    feat_data = torch.tensor(cluster_table.select(cluster_feat).to_pandas().to_numpy())
    min_vals = torch.tensor(list(min_feat_clusters.values()))
    max_vals = torch.tensor(list(max_feat_clusters.values()))
    feat_data = (feat_data - min_vals) / (max_vals - min_vals)
    # clamp needed if val/test data has min/max greater than train set min/max
    feat_data = torch.clamp(feat_data, min=0, max=1)
    data["clusters"].x = feat_data.float()  # might not need .float()

    # compute edge connections - use polars for loc table
    loc_table = pl.from_arrow(loc_table)

    # locs with clusterID connected to that cluster clusterID
    #group = loc_table.group_by("clusterID", maintain_order=True).agg(
    #    pl.col("x").agg_groups()
    #)
    #group = group.with_columns(
    #    pl.col("clusterID"), pl.col("x").list.len().alias("count")
    #)
    #count = group["count"].to_numpy()
    #clusterIDlist = [[i] * count[i] for i in range(len(count))]
    #group = group.with_columns(pl.Series("clusterIDlist", clusterIDlist))
    #loc_indices = group["x"].to_numpy()
    #cluster_indices = group["clusterIDlist"].to_numpy()
    #loc_indices_stack = np.concatenate(loc_indices, axis=0)
    #cluster_indices_stack = np.concatenate(cluster_indices, axis=0)
    #loc_cluster_edges = np.stack([loc_indices_stack, cluster_indices_stack])
    loc_cluster_edges = np.stack([np.arange(0, len(loc_table)), loc_table['clusterID']])

    # locs with same clusterID
    edges = []
    print(loc_cluster_edges[1])
    # iterate through clusters
    for i in range(np.max(loc_cluster_edges[1])):
        # get the loc indices for the clusters
        loc_indices = loc_cluster_edges[0][np.where(loc_cluster_edges[1]==i)]
        combos = list(itertools.combinations(loc_indices, 2))
        combos = np.ascontiguousarray(np.transpose(combos))
        edges.append(combos)
    loc_loc_edges = np.concatenate(edges, axis=-1)

    # knn on clusters
    x = torch.tensor(cluster_table["x_mean"].to_numpy())
    y = torch.tensor(cluster_table["y_mean"].to_numpy())
    coords = torch.stack([x, y], axis=-1)
    batch = torch.zeros(len(coords))
    cluster_cluster_edges = knn_graph(coords, k=kneighbours, batch=batch, loop=False)

    # edges in correct data format and undirected where relevant
    loc_loc_edges = loc_loc_edges.astype(int)
    loc_loc_edges = torch.from_numpy(loc_loc_edges)
    loc_loc_edges = to_undirected(loc_loc_edges)
    cluster_cluster_edges = to_undirected(cluster_cluster_edges)
    loc_cluster_edges = loc_cluster_edges.astype(int)
    loc_cluster_edges = torch.from_numpy(loc_cluster_edges)

    data["locs", "in", "cluster"].edge_index = loc_cluster_edges
    data["locs", "clusteredwith", "locs"].edge_index = loc_loc_edges
    data["clusters", "near", "clusters"].edge_index = cluster_cluster_edges

    warnings.warn("Need to check that graph is connected correctly")

    return data
