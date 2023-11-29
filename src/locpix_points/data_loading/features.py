""" Features module

Extracts features for the data
"""

import torch
import polars as pl
import numpy as np
import itertools
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, is_undirected, contains_self_loops, add_self_loops
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
    kneighboursclusters,
    fov_x,
    fov_y,
    kneighbourslocs=None,
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
        kneighboursclusters (int) : How many nearest neighbours to consider constructing knn graph for
            cluster dataset
        fov_x (float) : Size of fov (x) in units of data
        fov_y (float) : Size of fov (y) in units of data
        kneighbourslocs (int) : How many nearest neighbours to consider constructing knn graph for
            loc loc dataset. If None then connects all locs within each cluster. Default (None)
    Returns:
        data (torch_geometric data) : Data with
            position and feature for eacch node"""

    # load in positions
    x_locs = torch.tensor(loc_table["x"].to_numpy())
    y_locs = torch.tensor(loc_table["y"].to_numpy())
    
    # load in features
    if min_feat_locs is None:
        assert (loc_feat is None or (type(loc_feat) is list and len(loc_feat) == 0))
        data['locs'].x = None
    else:
        assert list(min_feat_locs.keys()) == loc_feat
        assert list(max_feat_locs.keys()) == loc_feat

        feat_data = torch.tensor(loc_table.select(loc_feat).to_pandas().to_numpy())
        min_vals = torch.tensor(list(min_feat_locs.values()))
        max_vals = torch.tensor(list(max_feat_locs.values()))
        feat_data = (feat_data - min_vals) / (max_vals - min_vals)
        # clamp needed if val/test data has min/max greater than train set min/max
        feat_data = torch.clamp(feat_data, min=0, max=1)
        data["locs"].x = feat_data.float()  # might not need .float()

    if min_feat_clusters is None:
        assert (cluster_feat is None or (type(cluster_feat) is list and len(cluster_feat) == 0))
        data['clusters'].x = None
    else:
        assert list(min_feat_clusters.keys()) == cluster_feat
        assert list(max_feat_clusters.keys()) == cluster_feat

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
    loc_cluster_edges = np.stack([np.arange(0, len(loc_table)), loc_table["clusterID"]])
    loc_cluster_edges = torch.from_numpy(loc_cluster_edges)
    #loc_cluster_edges = loc_cluster_edges.to(torch.int64)

    # knn on clusters
    x = torch.tensor(cluster_table["x_mean"].to_numpy())
    y = torch.tensor(cluster_table["y_mean"].to_numpy())
    # first need to check that clusterID is also in correct ordered from 0 to max cluster ID
    assert(np.array_equal(cluster_table['clusterID'].to_numpy(), np.arange(0,len(x))))
    coords = torch.stack([x, y], axis=-1)
    batch = torch.zeros(len(coords))
    # add 1 as with loop = True considers itself as one of the k neighbours
    cluster_cluster_edges = knn_graph(coords, k=kneighboursclusters+1, batch=batch, loop=True)
    cluster_cluster_edges = to_undirected(cluster_cluster_edges)
    
    # first need to check locs in correct form
    assert(np.array_equal(loc_cluster_edges[0].numpy(),np.arange(0, len(loc_cluster_edges[0]))))
    if kneighbourslocs is None:
        # locs with same clusterID
        edges = []
        # iterate through clusters
        for i in range(torch.max(loc_cluster_edges[1])):
            # get the loc indices for the clusters
            loc_indices = loc_cluster_edges[0][np.where(loc_cluster_edges[1] == i)]
            combos = list(itertools.permutations(loc_indices, 2))
            combos = np.ascontiguousarray(np.transpose(combos))
            edges.append(combos)
        loc_loc_edges = np.concatenate(edges, axis=-1)
        loc_loc_edges = torch.from_numpy(loc_loc_edges)
        loc_loc_edges, _ = add_self_loops(loc_loc_edges)
    else:
        batch_loc_loc = torch.tensor(loc_table['clusterID'].to_numpy())
        indices = torch.sort(batch_loc_loc).indices
        x_locs_idx = x_locs[indices]
        y_locs_idx = y_locs[indices]
        batch_loc_loc = batch_loc_loc[indices]
        loc_indices = np.arange(0, len(loc_table))[indices]
        loc_coords = torch.stack([x_locs_idx, y_locs_idx], axis=-1)
        # add 1 as with loop = True considers itself as one of the k neighbours
        loc_loc_edges = knn_graph(loc_coords, k=kneighbourslocs+1, batch=batch_loc_loc, loop=True)  
        # loc loc edges [0] is the neighbours
        # loc loc edges [1] is the original points
        locs_zero = loc_indices[loc_loc_edges[0]]
        locs_one = loc_indices[loc_loc_edges[1]]
        loc_loc_edges = np.stack([locs_zero, locs_one])
        loc_loc_edges = torch.from_numpy(loc_loc_edges)
    #loc_loc_edges = loc_loc_edges.to(torch.int64) 

    # Load in positions afterwards as scale data to between -1 and 1 - this might affect
    # above code
    # scale positions
    min_x = x_locs.min()
    min_y = y_locs.min()
    x_range = x_locs.max() - min_x
    y_range = y_locs.max() - min_y
    if x_range < 0.95*fov_x:
            warnings.warn(f"Range of x data: {x_range} is smaller than 95% of the wdith of the fov: {fov_x}")
    if y_range < 0.95*fov_y:
            warnings.warn(f"Range of y data: {y_range} is smaller than 95% of the height of the fov: {fov_y}")
    range_xy = max(x_range, y_range)

    # scale position
    # shift and scale biggest axis from -1 to 1
    x_locs = (x_locs - min_x)/range_xy 
    y_locs = (y_locs - min_y)/range_xy 
    # scale to between -1 and 1
    x_locs = 2.0*x_locs - 1.0
    y_locs = 2.0*y_locs - 1.0
    assert (x_locs.min() == -1.0 or y_locs.min() == 1.0)
    assert (x_locs.max() == 1.0 or y_locs.max() == 1.0)
    loc_coords = torch.stack((x_locs, y_locs), dim=1)
    data["locs"].pos = loc_coords.float()

    # scale cluster coordinates
    x_clusters = torch.tensor(cluster_table["x_mean"].to_numpy())
    y_clusters = torch.tensor(cluster_table["y_mean"].to_numpy())

    # scale from -1 to 1
    x_clusters = (x_clusters - min_x)/range_xy 
    y_clusters = (y_clusters - min_y)/range_xy 
    # scale from -1 to 1
    x_clusters = 2.0*x_clusters - 1.0
    y_clusters = 2.0*y_clusters - 1.0
    cluster_coords = torch.stack((x_clusters, y_clusters), dim=1)

    data["clusters"].pos = cluster_coords.float()

    data["locs", "in", "clusters"].edge_index = loc_cluster_edges
    data["locs", "clusteredwith", "locs"].edge_index = loc_loc_edges
    data["clusters", "near", "clusters"].edge_index = cluster_cluster_edges

    #warnings.warn(f'Loc to cluster edges are undirected: {is_undirected(loc_cluster_edges)}')
    #warnings.warn(f'Loc to loc edges are undirected: {is_undirected(loc_loc_edges)} and contains self loops: {contains_self_loops(loc_loc_edges)}')
    #warnings.warn(f'Cluster to cluster edges are undirected: {is_undirected(cluster_cluster_edges)} and contains self loops: {contains_self_loops(cluster_cluster_edges)}')

    #warnings.warn(f'1 unit in new space == {range_xy/2.0} in original units')
    #warnings.warn("Need to check that graph is connected correctly")
    #warnings.warn("Data should be normalised and scaled correctly")

    return data
