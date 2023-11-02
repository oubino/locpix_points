import polars as pl
import numpy as np

loc_table = pl.read_parquet('tests/output/preprocessed/featextract/locs/cancer_2.parquet')

locs_cluster_edges = np.stack([np.arange(0, len(loc_table)), loc_table['clusterID']])
print(locs_cluster_edges)
print(locs_cluster_edges.shape)

"""

print(loc_table)
group = loc_table.group_by("clusterID", maintain_order=True).agg(
    pl.col("x").agg_groups()
)
group = group.with_columns(
    pl.col("clusterID"), pl.col("x").list.len().alias("count")
)

count = group["count"].to_numpy()
clusterIDlist = [[i] * count[i] for i in range(len(count))]
group = group.with_columns(pl.Series("clusterIDlist", clusterIDlist))
loc_indices = group["x"].to_numpy()
cluster_indices = group["clusterIDlist"].to_numpy()
loc_indices_stack = np.concatenate(loc_indices, axis=0)
cluster_indices_stack = np.concatenate(cluster_indices, axis=0)
loc_cluster_edges = np.stack([loc_indices_stack, cluster_indices_stack])

"""