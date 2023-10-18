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

def cluster_data(df, eps=50.0, minpts=10, x_col='x', y_col='y'):
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
    dataframe['x'] = df[x_col].to_numpy()
    dataframe['y'] = df[y_col].to_numpy()
    dbscan = DBSCAN(eps = eps, min_samples = minpts)
    dbscan.fit(dataframe)

    raise ValueError("Is this correctly implemented need to check")
    df = df.with_columns(pl.lit(dbscan.labels_.to_numpy().astype('int32')).alias('clusterID'))
    # return the original df with cluster id for each loc 
    return df

def basic_cluster_feats(df, col_name='clusterID', x_name='x', y_name='y'):

    # take in loc df with cluster id per cluster
    cluster_df = df.groupby(col_name).agg([pl.count(), pl.col(x_name).mean().suffix('_mean'), pl.col(y_name).mean().suffix('_mean'), ((((pl.col(x_name)-pl.col(x_name).mean())**2).sum() + ((pl.col(y_name)-pl.col(y_name).mean())**2).sum())/pl.count()).alias('RGyration')])

    # number of rows = number of clusters
    raise ValueError("Check radius of gyration correctly calculated")

    return cluster_df

def pca_fn(X):
    dX = da.from_array(X, chunks=X.shape)
    pca = PCA(n_components=2)
    pca.fit(dX)
    components = pca.components_
    variance = pca.explained_variance_
    variance_ratio = pca.explained_variance_
    singular_values = pca.singular_values_
    raise ValueError('Which features of PCA do we want?')
    return singular_values

def pca_cluster(df, col_name='clusterID'):
    """Calculate pca for each cluster
    
    Args:
        df (polars df) : Input dataframe
        col_name (string) : Name for the cluster column
        
    Returns:
        cluster_df (polars df) : Dataframe detailing the cluster details"""
 
    df_split = df.partition_by(col_name)
    cluster_id = df[col_name].unique().to_numpy()

    array_list = [df.drop(col_name).to_numpy() for df in df_split] # slow

    lazy_results = []
    client = Client()

    for arr in array_list:
        lazy_result = dask.delayed(pca_fn)(arr)
        lazy_results.append(lazy_result)

    results = dask.compute(*lazy_results)

    cluster_df = pl.DataFrame({'clusterID':cluster_id, 'pca':results})

    raise ValueError("Is it normalised correctly? Need to check against known result?")

    return cluster_df

def convex_hull(array):
    """Convex hull function
    
    Args:
        array (numpy array) : Input array
        
    Returns:
        hull.area (float) : Perimeter of the 2D convex hull
        hull.volume (float) : Area of the 3D convex hull
        np.max(neigh_dist) : Maximum length of the convex hull"""
    hull = ConvexHull(array)
    vertices = hull.vertices
    neigh = NearestNeighbors(n_neighbors=len(vertices))
    neigh.fit(array[vertices])
    neigh_dist, _ = neigh.kneighbors(array[vertices], return_distance=True)
    return hull.area, hull.volume, np.max(neigh_dist)

def convex_hull_cluster(df, col_name='clusterID'):
    """Calculate convex hull for each cluster
    
    Args:
        df (polars df) : Input dataframe 
        col_name (string) : Name for the cluster column
        
    Returns:
        cluster_df (polars df) : Dataframe detailing the cluster details"""
    
    df_split = df.partition_by(col_name)
    cluster_id = df[col_name].unique().to_numpy()

    array_list = [df.drop(col_name).to_numpy() for df in df_split] # slow

    lazy_results = []
    client = Client()

    for arr in array_list:
        lazy_result = dask.delayed(convex_hull)(arr)
        lazy_results.append(lazy_result)

    results = dask.compute(*lazy_results)

    cluster_df = pl.DataFrame({'clusterID':cluster_id, 'convex_hull':results})

    raise ValueError("Is it normalised correctly? Need to check against known result?")

    return cluster_df


