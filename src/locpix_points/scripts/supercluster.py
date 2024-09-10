import os
import pyarrow.parquet as pq
import polars as pl
import numpy as np
from locpix_points.preprocessing import datastruc
import cudf
from cuml.cluster import DBSCAN, KMeans


folder = "preprocessed/featextract/clusters"
folder_locs = "preprocessed/featextract/locs"

folder_load = "preprocessed_all/featextract/clusters"
folder_locs_load = "preprocessed_all/featextract/locs"

clusters = []

type = input("DBSCAN (0) or KMEANS (1): ")
type = int(type)
if type not in [0, 1]:
    raise ValueError("0 or 1 please")
if type == 0:
    type = "DBSCAN"
elif type == 1:
    type = "KMEANS"

for file in os.listdir(folder):
    path = os.path.join(folder_load, file)
    path_loc = os.path.join(folder_locs_load, file)

    item_cluster = datastruc.item(None, None, None, None, None)
    item_cluster.load_from_parquet(path)

    item_loc = datastruc.item(None, None, None, None, None)
    item_loc.load_from_parquet(path_loc)

    df = item_cluster.df.drop("superclusterID")
    df = item_cluster.df.drop("superclusters_0")
    df = item_cluster.df.drop("superclusters_1")
    df = df.drop("superclusterID")

    # ---- generate superclusters 0 ----

    dataframe = cudf.DataFrame()
    dataframe["x"] = df["x_mean"].to_numpy()
    dataframe["y"] = df["y_mean"].to_numpy()

    clusters.append(len(df))

    if len(df) < 250:
        raise ValueError("Should not have fewer than 250 clusters")

    if type == "KMEANS":
        # n_clusters = int(df["clusterID"].max()/2)
        n_clusters = 250
        cluster_algo = KMeans(n_clusters=n_clusters)
    elif type == "DBSCAN":
        cluster_algo = DBSCAN(eps=750, min_samples=3)

    cluster_algo.fit(dataframe)
    df = df.with_columns(
        pl.lit(cluster_algo.labels_.to_numpy().astype("int32")).alias("superclusters_0")
    )
    df = df.filter(pl.col("superclusters_0") != -1)

    sc_0_df = df.group_by("superclusters_0").agg(
        [
            pl.col("x_mean").mean().alias("x_sc_0"),
            pl.col("y_mean").mean().alias("y_sc_0"),
        ]
    )
    df = df.join(sc_0_df, on="superclusters_0")

    # ---- generate superclusters 1 ----

    dataframe = cudf.DataFrame()
    df_mod = df[["superclusters_0", "x_sc_0", "y_sc_0"]].unique()
    dataframe["x"] = df_mod["x_sc_0"].to_numpy()
    dataframe["y"] = df_mod["y_sc_0"].to_numpy()

    if type == "KMEANS":
        # n_clusters = int(df["superclusters_0"].max()/2)
        n_clusters = 25
        cluster_algo = KMeans(n_clusters=n_clusters)
    elif type == "DBSCAN":
        cluster_algo = DBSCAN(eps=5000, min_samples=2)

    cluster_algo.fit(dataframe)
    df_mod = df_mod.with_columns(
        pl.lit(cluster_algo.labels_.to_numpy().astype("int32")).alias("superclusters_1")
    )

    df_mod = df_mod.drop(["x_sc_0", "y_sc_0"])

    if df_mod["superclusters_1"].max() < 2:
        print("Fewer than 3 superclusters_1")
        print(" ", item_cluster.name)
        print(" ", df_mod["superclusters_1"].max())
        if df_mod["superclusters_1"].max() == -1:
            raise ValueError("No superclusters 1")

    df = df.join(df_mod, on="superclusters_0")
    df = df.filter(pl.col("superclusters_1") != -1)
    df = df.drop(["x_sc_0", "y_sc_0"])

    # ---- reorder the clusters ----

    include_clusters = df["clusterID"]

    unique_clusters = list(df["superclusters_1"].unique(maintain_order=True))
    map = {value: i for i, value in enumerate(unique_clusters)}
    df = df.with_columns(
        pl.col("superclusters_1").map_dict(map).alias("superclusters_1")
    )
    df = df.sort("superclusters_1")

    unique_clusters = list(df["superclusters_0"].unique(maintain_order=True))
    map = {value: i for i, value in enumerate(unique_clusters)}
    df = df.with_columns(
        pl.col("superclusters_0").map_dict(map).alias("superclusters_0")
    )
    df = df.sort("superclusters_0")

    unique_clusters = list(df["clusterID"].unique(maintain_order=True))
    map_clusters = {value: i for i, value in enumerate(unique_clusters)}
    df = df.with_columns(pl.col("clusterID").map_dict(map_clusters).alias("clusterID"))
    df = df.sort("clusterID")
    item_cluster.df = df

    loc_df = item_loc.df.filter(pl.col("clusterID").is_in(include_clusters))
    loc_df = loc_df.with_columns(
        pl.col("clusterID").map_dict(map_clusters).alias("clusterID")
    )
    loc_df = loc_df.sort("clusterID")
    item_loc.df = loc_df

    # assert that they are ordered
    assert np.all(np.diff(item_loc.df["clusterID"]) >= 0)
    assert np.all(np.diff(item_cluster.df["clusterID"]) >= 0)
    assert np.all(np.diff(item_cluster.df["superclusters_0"]) >= 0)
    assert np.all(np.diff(item_cluster.df["superclusters_1"]) >= 0)

    item_cluster.save_to_parquet(
        folder,
        drop_zero_label=False,
        drop_pixel_col=False,
        overwrite=True,
    )

    item_loc.save_to_parquet(
        folder_locs,
        drop_zero_label=False,
        drop_pixel_col=False,
        overwrite=True,
    )

print("Minimum number of clusters", np.min(clusters))
print("Maximum number of clusters", np.max(clusters))
