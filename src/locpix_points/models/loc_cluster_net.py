"""LocClusterNet

Network embeds localisations within clusters.
Concatenates this embedding to user derived cluster features.
GraphNet operates on clusters, goes through linear layer, prediction!

Helped using https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage.py

"""

import torch
from torch_geometric.nn import (
    PointNetConv,
    HeteroConv,
    conv,
    MLP,
)
from torch_geometric.nn.pool import global_max_pool
from torch.nn import Linear


class LocEncoder(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        #raise ValueError("Need to define how many channels custom")
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x_locs, pos_locs, edge_index_dict):
        #raise ValueError("check + do we need a relu?")
        loc_x = self.conv(
            x_locs, pos_locs, edge_index_dict["locs", "clusteredwith", "locs"]
        ).relu()
        return loc_x


class Loc2Cluster(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #raise ValueError("Max or sum?")
        self.conv = HeteroConv(
            {("locs", "in", "clusters"): conv.SimpleConv(aggr="max")}, aggr=None
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        #raise ValueError("Do I need torch.squeeze")
        #raise ValueError("is dimension concatenating in correct")
        out["clusters"] = torch.squeeze(out["clusters"])
        x_dict["clusters"] = torch.cat((x_dict["clusters"], out["clusters"]), dim=-1)
        return x_dict["clusters"]


class ClusterEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #raise ValueError("Is number of channels correct, and max or average aggregate")
        self.conv = HeteroConv(
            {("clusters", "near", "clusters"): conv.SAGEConv(in_channels, out_channels)}, aggr="max"
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        #raise ValueError("Wrong axis when have batch")
        return out['clusters']


class LocClusterNet(torch.nn.Module):
    def __init__(self, foo):
        super().__init__()
        self.name = "loc_cluster_net"
        self.foo = foo
        # wrong input channel size 2 might change if locs have features
        self.loc_encode_0 = LocEncoder(MLP([4, 4, 4, 4]))
        self.loc_encode_1 = LocEncoder(MLP([6, 6, 6, 6]))
        self.loc_encode_2 = LocEncoder(MLP([8, 8, 8, 8]))
        self.loc2cluster = Loc2Cluster()
        self.clusterencoder_0 = ClusterEncoder(17,18)
        self.clusterencoder_1 = ClusterEncoder(18, 24)
        self.clusterencoder_2 = ClusterEncoder(24, 32)
        self.linear = Linear(32, 2)

    def forward(self, data):   

        x_dict = data.x_dict
        pos_dict = data.pos_dict
        edge_index_dict = data.edge_index_dict

        # embed each localisation 
        x_dict["locs"] = self.loc_encode_0(x_dict["locs"], pos_dict['locs'], edge_index_dict)
        x_dict["locs"] = self.loc_encode_1(x_dict["locs"], pos_dict['locs'], edge_index_dict)
        x_dict["locs"] = self.loc_encode_2(x_dict["locs"], pos_dict['locs'], edge_index_dict)

        # pool the embedding for each localisation to its cluster and concatenate this embedding with previous cluster embedding
        x_dict["clusters"] = self.loc2cluster(x_dict, edge_index_dict)

        # operate graph net on clusters, finish with 
        x_dict["clusters"] = self.clusterencoder_0(x_dict, edge_index_dict)
        x_dict["clusters"] = self.clusterencoder_1(x_dict, edge_index_dict)
        x_dict["clusters"] = self.clusterencoder_2(x_dict, edge_index_dict)

        # pooling step so end up with one feature vector per fov
        x_dict['clusters'] = global_max_pool(x_dict['clusters'], data['clusters'].batch)

        # linear layer on each fov feature vector
        output = self.linear(x_dict["clusters"]).log_softmax(dim=-1)

        return output
