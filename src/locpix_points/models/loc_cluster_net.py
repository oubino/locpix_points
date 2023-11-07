"""LocClusterNet

Network embeds localisations within clusters.
Concatenates this embedding to user derived cluster features.
GraphNet operates on clusters, goes through linear layer, prediction!

Helped using https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage.py

"""

import torch
from torch_geometric.nn import (
    SAGEConv,
    HeteroConv,
    conv,
)
from torch.nn import Linear


class LocEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #raise ValueError("Need to define how many channels custom")
        self.conv = SAGEConv((-1, -1), 7)

    def forward(self, x_dict, edge_index_dict):
        #raise ValueError("check + do we need a relu?")
        loc_x = self.conv(
            x_dict["locs"], edge_index_dict["locs", "clusteredwith", "locs"]
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
    def __init__(self):
        super().__init__()
        #raise ValueError("Is number of channels correct, and max or average aggregate")
        self.conv = HeteroConv(
            {("clusters", "near", "clusters"): conv.SAGEConv((-1, -1), 4)}, aggr="max"
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        #raise ValueError("Wrong axis when have batch")
        out, _ = torch.max(out["clusters"], axis=0)
        return out


class LocClusterNet(torch.nn.Module):
    def __init__(self, foo):
        super().__init__()
        self.name = "loc_cluster_net"
        self.foo = foo
        self.loc_encode = LocEncoder()
        self.loc2cluster = Loc2Cluster()
        self.clusterencoder = ClusterEncoder()
        self.linear = Linear(4, 1)

    def forward(self, x_dict, edge_index_dict):
        # embed each cluster, finish with pooling step
        x_dict["locs"] = self.loc_encode(x_dict, edge_index_dict)

        # for each cluster concatenate this embedding with previous state
        x_dict["clusters"] = self.loc2cluster(x_dict, edge_index_dict)

        # operate graph net on clusters, finish with pooling step
        x_dict["clusters"] = self.clusterencoder(x_dict, edge_index_dict)

        # linear layer
        output = self.linear(x_dict["clusters"])

        return output
