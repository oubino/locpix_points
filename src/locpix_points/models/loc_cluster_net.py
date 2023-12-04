"""LocClusterNet

Network embeds localisations within clusters.
Concatenates this embedding to user derived cluster features.
GraphNet operates on clusters, goes through linear layer, prediction!

Helped using https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage.py
&
https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing

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

    def forward(self, x_dict, edge_index_dict, cluster_feats_present=True):
        out = self.conv(x_dict, edge_index_dict)
        #raise ValueError("Do I need torch.squeeze")
        #raise ValueError("is dimension concatenating in correct")
        out["clusters"] = torch.squeeze(out["clusters"])
        if cluster_feats_present:
            x_dict["clusters"] = torch.cat((x_dict["clusters"], out["clusters"]), dim=-1)
        else:
            x_dict['clusters'] = out['clusters']
        return x_dict["clusters"]


class ClusterEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #raise ValueError("Is number of channels correct, and max or average aggregate")
        self.conv = HeteroConv(
            {("clusters", "near", "clusters"): conv.GINConv(in_channels, out_channels)}, aggr="max"
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict).relu()
        #raise ValueError("Wrong axis when have batch")
        return out['clusters']

class LocNet(torch.nn.Module):
    def __init__(self, encoder_0, encoder_1, encoder_2, loc2cluster, device):
        super().__init__()
        self.device = device
        self.encoder_0 = encoder_0
        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2
        self.loc2cluster = loc2cluster

    def forward(self, data):

        x_dict, pos_dict, edge_index_dict, cluster_feats_present = parse_data(data, self.device)

        x_dict["locs"] = self.encoder_0(x_dict["locs"], pos_dict['locs'], edge_index_dict)
        x_dict["locs"] = self.encoder_1(x_dict["locs"], pos_dict['locs'], edge_index_dict)
        x_dict["locs"] = self.encoder_2(x_dict["locs"], pos_dict['locs'], edge_index_dict)

        # pool the embedding for each localisation to its cluster and concatenate this embedding with previous cluster embedding
        x_dict["clusters"] = self.loc2cluster(x_dict, edge_index_dict, cluster_feats_present)

        return x_dict, pos_dict, edge_index_dict


class ClusterNet(torch.nn.Module):
    def __init__(self, cluster_encoder_0, cluster_encoder_1, cluster_encoder_2, linear):
        super().__init__()
        self.cluster_encoder_0 = cluster_encoder_0
        self.cluster_encoder_1 = cluster_encoder_1
        self.cluster_encoder_2 = cluster_encoder_2
        self.linear = linear
    
    def forward(self, x_dict, edge_index_dict, batch):

        x_dict["clusters"] = self.cluster_encoder_0(x_dict, edge_index_dict)
        x_dict["clusters"] = self.cluster_encoder_1(x_dict, edge_index_dict)
        x_dict["clusters"] = self.cluster_encoder_2(x_dict, edge_index_dict)

        # pooling step so end up with one feature vector per fov
        x_dict['clusters'] = global_max_pool(x_dict['clusters'], batch)

        # linear layer on each fov feature vector
        return self.linear(x_dict["clusters"])


def parse_data(data, device):

    try:
        x_dict = data.x_dict
        try:
            x_dict['locs']
        except KeyError:
            x_dict['locs'] = None
        try:
            x_dict['clusters']
            cluster_feats_present = True
        except KeyError:
            x_dict['clusters'] = torch.ones((data['clusters'].batch.shape[0], 1))
            cluster_feats_present = False  
    # neither locs nor clusters have features
    except KeyError:
        x_dict = {'locs': None, 'clusters': torch.ones((data['clusters'].batch.shape[0], 1), device=self.device)}
        cluster_feats_present = False
    
    pos_dict = data.pos_dict
    edge_index_dict = data.edge_index_dict

    return x_dict, pos_dict, edge_index_dict, cluster_feats_present

class LocClusterNet(torch.nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.name = "loc_cluster_net"
        # wrong input channel size 2 might change if locs have features
        self.loc_net = LocNet(LocEncoder(MLP(config['LocEncoderChannels'][0])),
                              LocEncoder(MLP(config['LocEncoderChannels'][1])),
                              LocEncoder(MLP(config['LocEncoderChannels'][2])),
                              Loc2Cluster(),
                              device
        )
        self.cluster_net = ClusterNet(
            ClusterEncoder(config['ClusterEncoderChannels'][0],config['ClusterEncoderChannels'][1]),
            ClusterEncoder(config['ClusterEncoderChannels'][1], config['ClusterEncoderChannels'][2]),
            ClusterEncoder(config['ClusterEncoderChannels'][2], config['ClusterEncoderChannels'][3]),
            Linear(config['ClusterEncoderChannels'][3], config['OutputChannels']),
        )

    def forward(self, data):   

        # embed each localisation
        x_dict, _, edge_index_dict = self.loc_net(data)

        # operate graph net on clusters, finish with 
        output = self.cluster_net(x_dict, edge_index_dict, data['clusters'].batch)

        return output.log_softmax(dim=-1)
