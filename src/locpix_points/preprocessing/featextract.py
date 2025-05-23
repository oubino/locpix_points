"""Feature extraction module.

This module contains functions to extract features from the data
"""

import math
import numpy as np
import polars as pl
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans


def cluster_data(
    df, eps=50.0, minpts=10, n_clusters=8, x_col="x", y_col="y", method="dbscan"
):
    """Cluster the data using DBSCAN or KMeans

    Args:
        df (polars df) : Input dataframe
        eps (float) : eps for DBSCAN
        minpts (int) : min samples for DBSCAN
        n_clusters (int) : num samples for KMeans
        x_col (string) : Name for the x column
        y_col (string) : Name for the y column
        method (string) : Choice of clustering method

    Returns:
        df (polars df) : Dataframe with additional column for cluster"""

    dataframe = pd.DataFrame()
    dataframe["x"] = df[x_col].to_numpy()
    dataframe["y"] = df[y_col].to_numpy()

    if method == "dbscan":
        dbscan = DBSCAN(eps=eps, min_samples=minpts)
        dbscan.fit(dataframe)

        df = df.with_columns(pl.lit(dbscan.labels_.astype("int32")).alias("clusterID"))
    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(dataframe)

        df = df.with_columns(pl.lit(kmeans.labels_.astype("int32")).alias("clusterID"))

    # return the original df with cluster id for each loc
    return df


def super_cluster(
    df,
    k=15,
):
    """Cluster the clusters into superclusters

    Args:
        df (polars df): Input dataframe
        k (int): Number of superclusters

    Returns:
        df (polars df): Output dataframe with superclusters

    Raises:
        ValueError: Temporary fix as need to re-write this function"""

    raise ValueError(
        "Incorporate the supercluster.py script and delete the supercluster.py script"
    )

    kmeans = KMeans(n_clusters=k)
    dataframe = cudf.DataFrame()
    dataframe["x"] = df["x_mean"].to_numpy()
    dataframe["y"] = df["y_mean"].to_numpy()
    kmeans.fit(dataframe)
    df = df.with_columns(
        pl.lit(kmeans.labels_.to_numpy().astype("int32")).alias("superclusterID")
    )

    return df


def basic_cluster_feats(df, col_name="clusterID", x_name="x", y_name="y"):
    """Calculate basic cluster features for the dataframe:
        locs per cluster, cluster COM, radius of gyration

    Args:
        df (pl.DataFrame): Dataframe containing the clusters
        col_name (string): Name of the column that identifies the clusters
        x_name (string): Name of the column that contains the x coords
            of the clusters
        y_name (string): Name of the column that contains the y coords
            of the clusters

    Returns:
        cluster_df (pl.DataFrame): Dataframe with cluters and the new features"""

    # take in loc df with cluster id per cluster
    cluster_df = df.group_by(col_name).agg(
        [
            pl.count(),
            pl.col(x_name).mean().alias(f"{x_name}_mean"),
            pl.col(y_name).mean().alias(f"{y_name}_mean"),
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
    """Calculates PCA for an array

    Args:
        X (array): Array to calculate PCA for

    Returns:
        linearity (float): Linearity for the cluster
        planarity (float): Planarity for the cluster
        length_pca (float): Length of the cluster according to PCA
        area_pca (float): Area of the cluster according to PCA
    """
    pca = PCA(n_components=2)
    pca.fit(X)
    # eigenvalues in order of size: variance[0], variance[1]
    variance = pca.explained_variance_
    # from 10.5194/isprsarchives-XXXVIII-5-W12-97-2011
    sigma_0 = math.sqrt(variance[0])
    sigma_1 = math.sqrt(variance[1])
    linearity = (sigma_0 - sigma_1) / sigma_0
    planarity = sigma_1 / sigma_0
    # as in 10.1073/pnas.0908971106
    # ratio between fwhm and s.d. is 2.35 therefore multiply sd by 2.35
    length_pca = 2.35 * sigma_0
    width_pca = 2.35 * sigma_1

    area_pca = length_pca * width_pca
    return linearity, planarity, length_pca, area_pca


def pca_cluster(df, x_col="x", y_col="y", col_name="clusterID"):
    """Calculate pca for each cluster

    Args:
        df (polars df) : Input dataframe
        x_col (string) : Name of the x column
        y_col (string) : Name of the y column
        col_name (string) : Name for the cluster column

    Returns:
        cluster_df (polars df) : Dataframe detailing the cluster details"""

    df_split = df.partition_by(col_name)
    cluster_id = df[col_name].unique().to_numpy()

    array_list = [
        df.select(pl.col([x_col, y_col])).to_numpy() for df in df_split
    ]  # slow

    results = []

    for arr in array_list:
        result = pca_fn(arr)
        results.append(result)

    array = np.array(results)
    linearities = array[:, 0]
    planarities = array[:, 1]
    lengths = array[:, 2]
    areas = array[:, 3]

    cluster_df = pl.DataFrame(
        {
            "clusterID": cluster_id,
            "linearity": linearities,
            "planarity": planarities,
            "length_pca": lengths,
            "area_pca": areas,
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
    # print("length via convex hull", length)
    return perimeter, area, length


def convex_hull_cluster(df, x_col="x", y_col="y", col_name="clusterID"):
    """Calculate convex hull for each cluster

    Args:
        df (polars df) : Input dataframe
        col_name (string) : Name for the cluster column
        x_col (string) : Name for the x column
        y_col (string) : Name for the y column

    Returns:
        cluster_df (polars df) : Dataframe detailing the cluster details"""

    df_split = df.partition_by(col_name)
    cluster_id = df[col_name].unique().to_numpy()

    array_list = [
        df.select(pl.col([x_col, y_col])).to_numpy() for df in df_split
    ]  # slow

    results = []

    for arr in array_list:
        result = convex_hull(arr)
        results.append(result)

    array = np.array(results)
    perimeters = array[:, 0]
    areas = array[:, 1]
    lengths = array[:, 2]

    cluster_df = pl.DataFrame(
        {
            "clusterID": cluster_id,
            "perimeter": perimeters,
            "area_convex_hull": areas,
            "length_convex_hull": lengths,
        }
    )

    return cluster_df
