# dbscan


import polars as pl
import numpy as np
import itertools
from torch_geometric.nn import (
    knn_graph,
)
import torch
from torch_geometric.data import HeteroData

from torch_geometric.nn import conv
from torch_geometric.utils import to_undirected
from torch_geometric.nn import (
    SAGEConv,
    HeteroConv,
    global_max_pool,
)
import warnings

#from scipy.spatial import ConvexHull
#import dask
#import dask.array as da
#from dask.distributed import Client
#from sklearn.neighbors import NearestNeighbors
#rng = np.random.default_rng()
#points = rng.random((30, 2))

def convex_hull(array):#

    hull = ConvexHull(array)
    vertices = hull.vertices
    neigh = NearestNeighbors(n_neighbors=len(vertices))
    neigh.fit(array[vertices])
    neigh_dist, _ = neigh.kneighbors(array[vertices], return_distance=True)
    return hull.area, np.max(neigh_dist)

def main():
    baa()

def baa():
    loc_table = pl.read_csv('test_loc_dataset.csv')
    cluster_table = pl.read_csv('test_cluster_dataset.csv')

    data = HeteroData()

    # load in positions
    x_locs = torch.tensor(loc_table["x"].to_numpy())
    y_locs = torch.tensor(loc_table["y"].to_numpy())
    loc_coords = torch.stack((x_locs, y_locs), dim=1)
    data['locs'].pos = loc_coords

    x_clusters = torch.tensor(cluster_table["x_mean"].to_numpy())
    y_clusters = torch.tensor(cluster_table["y_mean"].to_numpy())
    cluster_coords = torch.stack((x_clusters, y_clusters), dim=1)
    data['clusters'].pos = cluster_coords

    loc_feat = ['photons', 'sigma']
    cluster_feat = ['area', 'length', 'volume', 'x_mean', 'y_mean']

    # calculate min/max for each column of training data and save to config file
    def min_max(df, feats):
        min_df = df.select(pl.col(feats).min())
        max_df = df.select(pl.col(feats).max())
        min_vals = min_df.to_numpy()[0]
        max_vals = max_df.to_numpy()[0]

        min_vals = dict(zip(feats, min_vals))
        max_vals = dict(zip(feats, max_vals))

        return min_vals, max_vals
    
    min_feat_locs, max_feat_locs = min_max(loc_table, loc_feat)
    min_feat_clusters, max_feat_clusters = min_max(cluster_table, cluster_feat)

    feat_data = torch.tensor(loc_table.select(loc_feat).to_pandas().to_numpy())
    min_vals = torch.tensor(list(min_feat_locs.values()))
    max_vals = torch.tensor(list(max_feat_locs.values()))
    feat_data = (feat_data - min_vals)/(max_vals - min_vals)
    feat_data = torch.clamp(feat_data, min=0, max=1)
    data['locs'].x = feat_data.float()

    feat_data = torch.tensor(cluster_table.select(cluster_feat).to_pandas().to_numpy())
    min_vals = torch.tensor(list(min_feat_clusters.values()))
    max_vals = torch.tensor(list(max_feat_clusters.values()))
    print
    feat_data = (feat_data - min_vals)/(max_vals - min_vals)
    feat_data = torch.clamp(feat_data, min=0, max=1)
    data['clusters'].x = feat_data.float()

    # compute edge connections

    # locs with clusterID connected to that cluster clusterID
    group = loc_table.group_by('clusterID', maintain_order=True).agg(pl.col('x').agg_groups())
    group = group.with_columns(pl.col('clusterID'), pl.col("x").list.len().alias('count'))
    count = group['count'].to_numpy()
    clusterIDlist = [[i]*count[i] for i in range(len(count))]
    group = group.with_columns(
        pl.Series('clusterIDlist',clusterIDlist)
    )
    loc_indices = group['x'].to_numpy()
    cluster_indices = group['clusterIDlist'].to_numpy()
    loc_indices_stack = np.concatenate(loc_indices, axis=0)
    cluster_indices_stack = np.concatenate(cluster_indices, axis=0)
    loc_cluster_edges = np.stack([loc_indices_stack, cluster_indices_stack])

    # locs with same clusterID
    edges = []
    for a in loc_indices:
        combos = list(itertools.combinations(a, 2))
        combos = np.ascontiguousarray(np.transpose(combos))
        edges.append(combos)
    loc_loc_edges = np.concatenate(edges, axis=-1)

    # knn on clusters
    x = torch.tensor(cluster_table['x_mean'])
    y = torch.tensor(cluster_table['y_mean'])
    coords = torch.stack([x,y], axis=-1)
    batch = torch.zeros(len(coords))
    cluster_cluster_edges = knn_graph(coords, k=3, batch=batch, loop=False)

    loc_loc_edges = loc_loc_edges.astype(int)
    loc_loc_edges = torch.from_numpy(loc_loc_edges)
    loc_loc_edges = to_undirected(loc_loc_edges)

    cluster_cluster_edges = to_undirected(cluster_cluster_edges)

    loc_cluster_edges = loc_cluster_edges.astype(int)
    loc_cluster_edges = torch.from_numpy(loc_cluster_edges)

    data['locs','in','clusters'].edge_index = loc_cluster_edges
    data['locs', 'clusteredwith', 'locs'].edge_index = loc_loc_edges
    data['clusters','near','clusters'].edge_index =  cluster_cluster_edges

    # test model
    class LocEncoder(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = SAGEConv((-1, -1), 7)

        def forward(self, x_dict, edge_index_dict):

            loc_x = self.conv(
                x_dict['locs'],
                edge_index_dict['locs', 'clusteredwith', 'locs']
            ).relu()
            return loc_x

    class Loc2Cluster(torch.nn.Module):

        def __init__(self):
            #super().__init__(aggr='max')
            super().__init__()

            self.conv = HeteroConv({
                ('locs', 'in', 'clusters'): conv.SimpleConv(aggr='max')
            }, aggr=None)

        def forward(self, x_dict, edge_index_dict):

            out = self.conv(
                x_dict, edge_index_dict
            )
            out['clusters'] = torch.squeeze(out['clusters'])
            x_dict['clusters'] = torch.cat((x_dict['clusters'], out['clusters']), dim=-1)

            return x_dict['clusters']
        
    class ClusterEncoder(torch.nn.Module):

        def __init__(self):
            #super().__init__(aggr='max')
            super().__init__()

            self.conv = HeteroConv({
                ('clusters', 'near', 'clusters'): conv.SAGEConv((-1, -1), 4)
            }, aggr='max')

        def forward(self, x_dict, edge_index_dict):
            
            print(x_dict['clusters'].shape)
            out = self.conv(
                x_dict, edge_index_dict
            )
            print(out.keys())
            print(out['clusters'].shape)

            print(out['clusters'])

            warnings.warn('This will be wrong axis when have batch')
            out, _ = torch.max(out['clusters'], axis=0)

            print(out)

            print(out.shape)

    class LocClusterNet(torch.nn.Module):

        #def __init__(self, name):
        #    self.name = name
        #    self.loc_embed = 
        def __init__(self):
            super().__init__()
            self.loc_encode = LocEncoder()
            self.loc2cluster = Loc2Cluster()
            self.clusterencoder = ClusterEncoder()

        def forward(self, x_dict, edge_index_dict):
            
            z_dict = {}

            # embed each cluster, finish with pooling step
            x_dict['locs'] = self.loc_encode(x_dict, edge_index_dict)

            # for each cluster concatenate this embedding with previous state 
            x_dict['clusters'] = self.loc2cluster(x_dict, edge_index_dict)

            # operate graph net on clusters, finish with pooling step
            x_dict['clusters'] = self.clusterencoder(x_dict, edge_index_dict)
            
            # linear layer

    model = LocClusterNet()
    
    out = model(
        data.x_dict,
        data.edge_index_dict
    )


    
def foo():

    col_name = 'cluster'
    df = pl.read_csv('test_output.csv')
    df_split = df.partition_by(col_name)
    cluster_id = df[col_name].unique().to_numpy()
    array_list = [df.drop(col_name).to_numpy() for df in df_split] # slow

    lazy_results = []
    client = Client()

    for arr in array_list:
        lazy_result = dask.delayed(convex_hull)(arr)
        lazy_results.append(lazy_result)

    results = dask.compute(*lazy_results)

    print(results)
    array = np.array(results)
    areas = array[:,0]
    lengths = array[:,1]

    cluster_df = pl.DataFrame({'clusterID':cluster_id, 'area':areas, 'length': lengths})
    print(cluster_df.columns)

    if 'lengtha' in cluster_df.columns:
        print('yes')


if __name__ == "__main__":
    main()
