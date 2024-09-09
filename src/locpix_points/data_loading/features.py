""" Features module

Extracts features for the data
"""

import itertools
import logging

import numpy as np
import polars as pl
import torch
from torch_geometric.nn import knn_graph, aggr
from torch_geometric.utils import (
    add_self_loops,
    to_undirected,
)


def sort_edge_index(x):
    idx = torch.sort(x[1]).indices

    x[0] = x[0][idx]
    x[1] = x[1][idx]

    idx = torch.sort(x[0], stable=True).indices

    x[0] = x[0][idx]
    x[1] = x[1][idx]

    return x


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

    loc_table = pl.from_arrow(loc_table)
    cluster_table = pl.from_arrow(cluster_table)
    loc_table = loc_table.sort("clusterID")

    # load in positions
    x_locs = torch.tensor(loc_table["x"].to_numpy())
    y_locs = torch.tensor(loc_table["y"].to_numpy())

    # load in features
    if min_feat_locs is None:
        assert loc_feat is None or (type(loc_feat) is list and len(loc_feat) == 0)
        data["locs"].x = None
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
        assert cluster_feat is None or (
            type(cluster_feat) is list and len(cluster_feat) == 0
        )
        data["clusters"].x = None
    else:
        assert list(min_feat_clusters.keys()) == cluster_feat
        assert list(max_feat_clusters.keys()) == cluster_feat

        feat_data = torch.tensor(
            cluster_table.select(cluster_feat).to_pandas().to_numpy()
        )
        min_vals = torch.tensor(list(min_feat_clusters.values()))
        max_vals = torch.tensor(list(max_feat_clusters.values()))
        feat_data = (feat_data - min_vals) / (max_vals - min_vals)
        # clamp needed if val/test data has min/max greater than train set min/max
        feat_data = torch.clamp(feat_data, min=0, max=1)
        data["clusters"].x = feat_data.float()  # might not need .float()

    # locs with clusterID connected to that cluster clusterID
    loc_cluster_edges = np.stack([np.arange(0, len(loc_table)), loc_table["clusterID"]])
    loc_cluster_edges = torch.from_numpy(loc_cluster_edges)
    # loc_cluster_edges = loc_cluster_edges.to(torch.int64)

    # knn on clusters
    x = torch.tensor(cluster_table["x_mean"].to_numpy())
    y = torch.tensor(cluster_table["y_mean"].to_numpy())
    # first need to check that clusterID is also in correct ordered from 0 to max cluster ID
    assert np.array_equal(cluster_table["clusterID"].to_numpy(), np.arange(0, len(x)))
    coords = torch.stack([x, y], axis=-1)
    batch = torch.zeros(len(coords))
    # add 1 as with loop = True considers itself as one of the k neighbours
    cluster_cluster_edges = knn_graph(
        coords, k=kneighboursclusters + 1, batch=batch, loop=True
    )
    cluster_cluster_edges = to_undirected(cluster_cluster_edges)

    # first need to check locs in correct form
    assert np.array_equal(
        loc_cluster_edges[0].numpy(), np.arange(0, len(loc_cluster_edges[0]))
    )
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
        batch_loc_loc = torch.tensor(loc_table["clusterID"].to_numpy())
        # loc_indices = np.arange(0, len(loc_table))[indices]
        loc_coords = torch.stack([x_locs, y_locs], axis=-1)
        # add 1 as with loop = True considers itself as one of the k neighbours
        loc_loc_edges = knn_graph(
            loc_coords, k=kneighbourslocs + 1, batch=batch_loc_loc, loop=True
        )
    # loc_loc_edges = loc_loc_edges.to(torch.int64)

    # Load in positions afterwards as scale data to between -1 and 1 - this might affect
    # above code
    # scale positions
    min_x = x_locs.min()
    min_y = y_locs.min()
    x_range = x_locs.max() - min_x
    y_range = y_locs.max() - min_y
    if x_range < 0.95 * fov_x:
        logging.info(
            f"Range of x data: {x_range} is smaller than 95% of the width of the fov: {fov_x}"
        )
    if y_range < 0.95 * fov_y:
        logging.info(
            f"Range of y data: {y_range} is smaller than 95% of the height of the fov: {fov_y}"
        )
    range_xy = max(x_range, y_range)

    # scale position
    # shift and scale biggest axis from -1 to 1
    x_locs = (x_locs - min_x) / range_xy
    y_locs = (y_locs - min_y) / range_xy
    # scale to between -1 and 1
    x_locs = 2.0 * x_locs - 1.0
    y_locs = 2.0 * y_locs - 1.0
    assert x_locs.min() == -1.0 or y_locs.min() == 1.0
    assert x_locs.max() == 1.0 or y_locs.max() == 1.0
    loc_coords = torch.stack((x_locs, y_locs), dim=1)
    data["locs"].pos = loc_coords.float()

    # scale cluster coordinates
    x_clusters = torch.tensor(cluster_table["x_mean"].to_numpy())
    y_clusters = torch.tensor(cluster_table["y_mean"].to_numpy())

    # scale from -1 to 1
    x_clusters = (x_clusters - min_x) / range_xy
    y_clusters = (y_clusters - min_y) / range_xy
    # scale from -1 to 1
    x_clusters = 2.0 * x_clusters - 1.0
    y_clusters = 2.0 * y_clusters - 1.0
    cluster_coords = torch.stack((x_clusters, y_clusters), dim=1)
    data["clusters"].pos = cluster_coords.float()

    #  ---- superclusters_0 ----
    data, x_sc_0, y_sc_0, cluster_id_sc_0 = supercluster_ID(
        data,
        cluster_table,
        x_clusters,
        y_clusters,
        torch.tensor(cluster_table["clusterID"].to_numpy(), dtype=torch.int64),
        "clusters",
        "superclusters_0",
    )

    # ---- superclusters_1 ----
    data, _, _, cluster_id_above = supercluster_ID(
        data,
        cluster_table,
        x_sc_0,
        y_sc_0,
        cluster_id_sc_0,
        "superclusters_0",
        "superclusters_1",
    )

    data["locs", "in", "clusters"].edge_index = loc_cluster_edges
    # loc_loc_edges = sort_edge_index(loc_loc_edges)
    data["locs", "clusteredwith", "locs"].edge_index = loc_loc_edges
    data["clusters", "near", "clusters"].edge_index = cluster_cluster_edges

    # warnings.warn(f'Loc to cluster edges are undirected: {is_undirected(loc_cluster_edges)}')
    # warnings.warn(f'Loc to loc edges are undirected: {is_undirected(loc_loc_edges)} and contains self loops: {contains_self_loops(loc_loc_edges)}')
    # warnings.warn(f'Cluster to cluster edges are undirected: {is_undirected(cluster_cluster_edges)} and contains self loops: {contains_self_loops(cluster_cluster_edges)}')

    # warnings.warn(f'1 unit in new space == {range_xy/2.0} in original units')
    # warnings.warn("Need to check that graph is connected correctly")
    # warnings.warn("Data should be normalised and scaled correctly")
    data.validate(raise_on_error=True)

    return data


def supercluster_ID(
    data,
    cluster_table,
    x_sub,
    y_sub,
    cluster_id_sub,
    subcluster_string,
    supercluster_string,
):
    """Assign supercluster to data point

    Args:
        data (pyg data): Dataitem from PyG
        cluster_table (dataframe): Dataframe with the superclusters in it
        x_sub (tensor): x coordinates of the cluster being clustered into supercluster
        y_sub (tensor): y coordinates of the cluster being clustered into supercluster
        cluster_id_sub (tensor): cluster ID for the clusters being clustered into supercluster
        subcluster_string (string): name of the subcluster
        supercluster_string (string): name of the supercluster in the dataset

    Returns:
        data (pyG dataitem): Dataitem with added supercluster
        x_super_clusters (tensor): x-coords of supercluster
        y_super_clusters (tensor): y-coords of supercluster
        superclusterID (tensor): ID for the supercluster
    """

    mean_aggr = aggr.MeanAggregation()

    # get the superclusterID from the dataframe
    superclusterID = torch.tensor(
        cluster_table[supercluster_string].to_numpy(), dtype=torch.long
    )
    # mean agg is misleading here - what we are doing is shrinking the list so we go
    # from
    # cluster_id_sub, superclusterID
    # 0, 0
    # 0, 0
    # 1, 1
    # 2, 2
    # 3, 2
    # to
    # cluster_id_sub, superclusterID
    # 0, 0
    # 1, 1
    # 2, 2
    # 3, 2
    # therfoer can take mean
    superclusterID = mean_aggr(superclusterID, index=cluster_id_sub, dim=0).to(
        torch.int64
    )

    # calculate the x, y positions of the super clusters
    x_super_clusters = mean_aggr(x_sub, index=superclusterID, dim=0)
    y_super_clusters = mean_aggr(y_sub, index=superclusterID, dim=0)
    super_cluster_coords = torch.stack((x_super_clusters, y_super_clusters), dim=1)

    # assign to the dataitem
    data[supercluster_string].index = superclusterID
    data[supercluster_string].pos = super_cluster_coords.float()

    # calculate edges
    sub_cluster_super_cluster_edges = np.stack(
        [np.arange(0, len(superclusterID)), superclusterID]
    )

    sub_cluster_super_cluster_edges = torch.from_numpy(sub_cluster_super_cluster_edges)
    data[
        subcluster_string, "in", supercluster_string
    ].edge_index = sub_cluster_super_cluster_edges

    return data, x_super_clusters, y_super_clusters, superclusterID


def load_loc(
    data,
    loc_table,
    loc_feat,
    min_feat_locs,
    max_feat_locs,
    fov_x,
    fov_y,
    kneighbours=None,
):
    """Load in position, features and edge index to each node

    Args:
        data (torch_geometric data) : Data item to
            load position, features and edge index into
        loc_table (parquet arrow table) : Localisation data
            in form of parquet file
        loc_feat (list) : List of features to consider in localisation dataset
        min_feat_locs (dict) : Minimum values of features for the locs training dataset
        max_feat_locs (dict) : Maxmimum values of features over locs training dataset
        fov_x (float) : Size of fov (x) in units of data
        fov_y (float) : Size of fov (y) in units of data
        kneighbours (int) : How many nearest neighbours to consider constructing knn graph for
            loc loc dataset. If None then no edges between locs Default (None)

    Returns:
        data (torch_geometric data) : Data with
            position and feature for eacch node"""

    loc_table = pl.from_arrow(loc_table)

    # load in positions
    x_locs = torch.tensor(loc_table["x"].to_numpy())
    y_locs = torch.tensor(loc_table["y"].to_numpy())

    # load in features
    if min_feat_locs is None:
        assert loc_feat is None or (type(loc_feat) is list and len(loc_feat) == 0)
        data.x = None
    else:
        assert list(min_feat_locs.keys()) == loc_feat
        assert list(max_feat_locs.keys()) == loc_feat

        feat_data = torch.tensor(loc_table.select(loc_feat).to_pandas().to_numpy())
        min_vals = torch.tensor(list(min_feat_locs.values()))
        max_vals = torch.tensor(list(max_feat_locs.values()))
        feat_data = (feat_data - min_vals) / (max_vals - min_vals)
        # clamp needed if val/test data has min/max greater than train set min/max
        feat_data = torch.clamp(feat_data, min=0, max=1)
        data.x = feat_data.float()  # might not need .float()

    if kneighbours is not None:
        batch = torch.zeros(len(data["locs"].x))
        loc_coords = torch.stack([x_locs, y_locs], axis=-1)
        # add 1 as with loop = True considers itself as one of the k neighbours
        loc_loc_edges = knn_graph(loc_coords, k=kneighbours + 1, batch=batch, loop=True)
        data.edge_index = loc_loc_edges

    # Load in positions afterwards as scale data to between -1 and 1 - this might affect
    # above code
    # scale positions
    min_x = x_locs.min()
    min_y = y_locs.min()
    x_range = x_locs.max() - min_x
    y_range = y_locs.max() - min_y
    if x_range < 0.95 * fov_x:
        logging.info(
            f"Range of x data: {x_range} is smaller than 95% of the width of the fov: {fov_x}"
        )
    if y_range < 0.95 * fov_y:
        logging.info(
            f"Range of y data: {y_range} is smaller than 95% of the height of the fov: {fov_y}"
        )
    range_xy = max(x_range, y_range)

    # scale position
    # shift and scale biggest axis from -1 to 1
    x_locs = (x_locs - min_x) / range_xy
    y_locs = (y_locs - min_y) / range_xy
    # scale to between -1 and 1
    x_locs = 2.0 * x_locs - 1.0
    y_locs = 2.0 * y_locs - 1.0
    assert x_locs.min() == -1.0 or y_locs.min() == 1.0
    assert x_locs.max() == 1.0 or y_locs.max() == 1.0
    loc_coords = torch.stack((x_locs, y_locs), dim=1)
    data.pos = loc_coords.float()

    data.validate(raise_on_error=True)

    return data
