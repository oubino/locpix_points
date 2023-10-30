"""Feature extraction module.

This module contains functions to extract features from the data
"""

import dask
import dask.array as da
from dask.distributed import Client
from dask_ml.decomposition import PCA
import polars as pl
from scipy.spatial import ConvexHull
from cuml.cluster import DBSCAN
import cudf
from sklearn.neighbors import NearestNeighbors
import numpy as np


def cluster_data(df, eps=50.0, minpts=10, x_col="x", y_col="y"):
    """Cluster the data using DBSCAN

    Args:
        df (polars df) : Input dataframe
        eps (float) : eps for DBSCAN
        minpts (int) : min samples for DBSCAN
        x_col (string) : Name for the x column
        y_col (string) : Name for the y column

    Returns:
        df (polars df) : Dataframe with additional column for cluster"""

    dataframe = cudf.DataFrame()
    dataframe["x"] = df[x_col].to_numpy()
    dataframe["y"] = df[y_col].to_numpy()
    dbscan = DBSCAN(eps=eps, min_samples=minpts)
    dbscan.fit(dataframe)

    df = df.with_columns(
        pl.lit(dbscan.labels_.to_numpy().astype("int32")).alias("clusterID")
    )
    # return the original df with cluster id for each loc
    return df


def basic_cluster_feats(df, col_name="clusterID", x_name="x", y_name="y"):
    # take in loc df with cluster id per cluster
    cluster_df = df.groupby(col_name).agg(
        [
            pl.count(),
            pl.col(x_name).mean().suffix("_mean"),
            pl.col(y_name).mean().suffix("_mean"),
            (
                (
                    ((pl.col(x_name) - pl.col(x_name).mean()) ** 2).sum()
                    + ((pl.col(y_name) - pl.col(y_name).mean()) ** 2).sum()
                )
                / pl.count()
            ).alias("RGyration"),
        ]
    )

    return cluster_df


def pca_fn(X):
    dX = da.from_array(X, chunks=X.shape)
    pca = PCA(n_components=2)
    pca.fit(dX)
    # eigenvalues in order of size: variance[0], variance[1] 
    variance = pca.explained_variance_
    # from 10.5194/isprsarchives-XXXVIII-5-W12-97-2011
    linearity = (variance[0] - variance[1])/variance[0]
    planarity = variance[1]/variance[0]
    #components = pca.components_
    #variance_ratio = pca.explained_variance_ratio
    #singular_values = pca.singular_values_
    # trial pca length: 99.7% data falls within 3x S.D of 
    # mean and x2 to get both sides of distribution
    # = 6 * vairance[0]
    length_pca = 6*variance[0]
    width_pca = 6*variance[1]

    print('pca length', length_pca)
    print('pca area', length_pca*width_pca)
    test = variance[2]
    raise ValueError('Above should fail')
    return linearity, planarity


def pca_cluster(df, col_name="clusterID"):
    """Calculate pca for each cluster

    Args:
        df (polars df) : Input dataframe
        col_name (string) : Name for the cluster column

    Returns:
        cluster_df (polars df) : Dataframe detailing the cluster details"""

    df_split = df.partition_by(col_name)
    cluster_id = df[col_name].unique().to_numpy()

    array_list = [df.drop(col_name).to_numpy() for df in df_split]  # slow

    lazy_results = []
    _ = Client()

    for arr in array_list:
        lazy_result = dask.delayed(pca_fn)(arr)
        lazy_results.append(lazy_result)

    results = dask.compute(*lazy_results)

    array = np.array(results)
    linearities = array[:, 0]
    planarities = array[:, 1]

    cluster_df = pl.DataFrame(
        {
            "clusterID": cluster_id,
            "linearity": linearities,
            "planarity":planarities,
        }
    )

    return cluster_df


def convex_hull(array):
    """Convex hull function

    Args:
        array (numpy array) : Input array

    Returns:
        perimieter (float) : Perimeter of the 2D convex hull
        area (float) : Area of the convex hull
        np.max(neigh_dist) : Maximum length of the convex hull"""

    hull = ConvexHull(array)
    vertices = hull.vertices
    neigh = NearestNeighbors(n_neighbors=len(vertices))
    neigh.fit(array[vertices])
    neigh_dist, _ = neigh.kneighbors(array[vertices], return_distance=True)
    perimeter = hull.area
    area = hull.volume
    length = np.max(neigh_dist)
    print('length via convex hull', length)
    input('stop after convex hull length')
    return perimeter, area, length


def convex_hull_cluster(df, col_name="clusterID"):
    """Calculate convex hull for each cluster

    Args:
        df (polars df) : Input dataframe
        col_name (string) : Name for the cluster column

    Returns:
        cluster_df (polars df) : Dataframe detailing the cluster details"""

    df_split = df.partition_by(col_name)
    cluster_id = df[col_name].unique().to_numpy()

    array_list = [df.drop(col_name).to_numpy() for df in df_split]  # slow

    lazy_results = []
    _ = Client()

    for arr in array_list:
        lazy_result = dask.delayed(convex_hull)(arr)
        lazy_results.append(lazy_result)

    results = dask.compute(*lazy_results)
    array = np.array(results)
    perimeters = array[:, 0]
    areas = array[:, 1]
    lengths = array[:, 2]

    cluster_df = pl.DataFrame(
        {"clusterID": cluster_id, "perimeter":perimeters, "area": areas, "length": lengths}
    )

    return cluster_df
