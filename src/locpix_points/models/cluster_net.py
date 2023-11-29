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
    HeteroConv,
    conv,
)
from torch_geometric.nn.pool import global_max_pool
from torch.nn import Linear


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


class ClusterNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "cluster_net"
        self.clusterencoder_0 = ClusterEncoder(config['ClusterEncoderChannels'][0],config['ClusterEncoderChannels'][1])
        self.clusterencoder_1 = ClusterEncoder(config['ClusterEncoderChannels'][1], config['ClusterEncoderChannels'][2])
        self.clusterencoder_2 = ClusterEncoder(config['ClusterEncoderChannels'][2], config['ClusterEncoderChannels'][3])
        self.linear = Linear(config['ClusterEncoderChannels'][3], config['OutputChannels'])

    def forward(self, data):   

        x_dict = data.x_dict

        pos_dict = data.pos_dict
        edge_index_dict = data.edge_index_dict

        # operate graph net on clusters, finish with 
        x_dict["clusters"] = self.clusterencoder_0(x_dict, edge_index_dict)
        x_dict["clusters"] = self.clusterencoder_1(x_dict, edge_index_dict)
        x_dict["clusters"] = self.clusterencoder_2(x_dict, edge_index_dict)

        # pooling step so end up with one feature vector per fov
        x_dict['clusters'] = global_max_pool(x_dict['clusters'], data['clusters'].batch)

        # linear layer on each fov feature vector
        output = self.linear(x_dict["clusters"]).log_softmax(dim=-1)

        return output
