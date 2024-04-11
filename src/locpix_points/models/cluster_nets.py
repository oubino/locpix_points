"""LocClusterNet

Network embeds localisations within clusters.
Concatenates this embedding to user derived cluster features.
GraphNet operates on clusters, goes through linear layer, prediction!

Helped using https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage.py
&
https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing

"""

import torch
from torch.nn import Linear
from torch_geometric.nn import MLP, HeteroConv, PointNetConv, conv
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from .point_transformer import PointTransformerEmbedding
from .point_net import PointNetEmbedding


class ClusterEncoder(torch.nn.Module):
    """Module for encoding clusters

    Attributes:
        channel_list (list): Channel sizes for the MLP used in the neural network
            used in Conv
        dropout (int): Dropout to apply to MLP
        conv_type (str): Either gin or transformer for GinConv or Transformer respectively
        channel_list (list): Channels for MLP if using GinConv
        out_channels (int): Out channels for TransformerConv
        heads (int): Number of multihead attention heads for TransformerConv
        concat (bool): If true concatenate features from heads, if false then average
        beta (bool): See pytorch geometric for more details"""

    def __init__(
        self,
        dropout,
        conv_type="gin",
        channel_list=None,
        out_channels=None,
        heads=1,
        concat=True,
        beta=False,
    ):
        super().__init__()
        # note here as all the same relation the aggr has no effect
        if conv_type == "gin":
            nn = MLP(channel_list, plain_last=False, dropout=dropout)
            self.conv = HeteroConv(
                {("clusters", "near", "clusters"): conv.GINConv(nn)}, aggr="max"
            )
        elif conv_type == "transformer":
            self.conv = HeteroConv(
                {
                    ("clusters", "near", "clusters"): conv.TransformerConv(
                        -1,
                        out_channels,
                        heads=heads,
                        concat=concat,
                        beta=beta,
                        dropout=dropout,
                    )
                },
                aggr="sum",
            )
        else:
            raise ValueError("conv_type should be gin or transformer")

    def forward(self, x_dict, edge_index_dict):
        """The method called when the encoder is used on a data item

        Args:
            x_dict (dict): Features of the locs/clusters
            edge_index_dict (dict): Edge connections between
                locs/clusters

        Returns:
            out["clusters"].relu() (torch.tensor): Encoded cluster features
        """
        out = self.conv(x_dict, edge_index_dict)
        return out["clusters"]
        # raise ValueError("Wrong axis when have batch")


class ClusterNet(torch.nn.Module):
    """Network for taking the cluster embeddings and making a classification

    Attributes:
        cluster_encoder_0 (torch.nn.module): First encoder for the clusters
        cluster_encoder_1 (torch.nn.module): Second encoder for the clusters
        cluster_encoder_2 (torch.nn.module): Third encoder for the clusters
        linear (torch.nn.module): Linear layer that operates on cluster embeddings
            and returns a classification
    """

    def __init__(self, cluster_encoder_0, cluster_encoder_1, cluster_encoder_2, linear):
        super().__init__()
        self.cluster_encoder_0 = cluster_encoder_0
        self.cluster_encoder_1 = cluster_encoder_1
        self.cluster_encoder_2 = cluster_encoder_2
        self.linear = linear

    def forward(self, x_dict, edge_index_dict, batch):
        """The method called when ClusterNet is used on a dataitem

        Args:
            x_dict (dict): dictionary with the features for the
                clusters/locs
            edge_index_dict (dict): contains the edge connections
                between the locs/clusters
            batch (torch.tensor): batch for the clusters

        Returns:
            self.linear(x_dict['clusters']): Log-probability for the classes
                for that FOV
        """
        x_dict["clusters"] = self.cluster_encoder_0(x_dict, edge_index_dict)
        x_dict["clusters"] = self.cluster_encoder_1(x_dict, edge_index_dict)
        x_dict["clusters"] = self.cluster_encoder_2(x_dict, edge_index_dict)

        # pooling step so end up with one feature vector per fov
        x_dict["clusters"] = global_max_pool(x_dict["clusters"], batch)

        # linear layer on each fov feature vector
        return self.linear(x_dict["clusters"])


def parse_data(data, device):
    """Parses a data item into the respective components
    x_dict, pos_dict, edge_index_dict and also returns whether
    there are cluster features present

    Args:
        data (torch.tensor): Data item to be parsed
        device (torch.device): Device to put the data on

    Returns:
        x_dict (dict): dictionary containing the features for
            the data
        pos_dict (dict): dictionary containing the positions for
            the data
        edge_index_dict (dict): dictionary containing the edge
            connections for the data
        cluster_feats_present (bool): Whether the clusters have
            features
    """
    try:
        x_dict = data.x_dict
        try:
            x_dict["locs"]
        except KeyError:
            x_dict["locs"] = None
        try:
            x_dict["clusters"]
            cluster_feats_present = True
        except KeyError:
            # num_clusters = data.pos_dict["clusters"].shape[0]
            x_dict["clusters"] = None  # torch.ones((num_clusters, 1), device=device)
            cluster_feats_present = False
    # neither locs nor clusters have features
    except KeyError:
        num_clusters = data.pos_dict["clusters"].shape[0]
        x_dict = {
            "locs": None,
            "clusters": None,  # torch.ones((num_clusters, 1), device=device),
        }
        cluster_feats_present = False

    pos_dict = data.pos_dict
    edge_index_dict = data.edge_index_dict

    return x_dict, pos_dict, edge_index_dict, cluster_feats_present


class ClusterNetHomogeneous(torch.nn.Module):
    """ClusterNetwork for a homogeneous graph instantiated from a heterogeneous graph
    This is used for PGExplainer which expects unnormalized class score therefore last layer
    is unnormalised class score

    Attributes:
        cluster_net_hetero (torch.nn.Module): The heterogeneous ClusterNetwork from which
            we will instantiate a homogeneous one
        config (dict): Configuration for the Network"""

    def __init__(self, cluster_net_hetero, config):
        super().__init__()

        if config["conv_type"] == "gin":
            # first
            self.cluster_encoder_0 = conv.GINConv(
                MLP(
                    config["ClusterEncoderChannels"][0],
                    plain_last=False,
                    dropout=config["dropout"],
                )
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_0.state_dict().items()
            }
            self.cluster_encoder_0.load_state_dict(state_dict_saved)
            # second
            self.cluster_encoder_1 = conv.GINConv(
                MLP(
                    config["ClusterEncoderChannels"][1],
                    plain_last=False,
                    dropout=config["dropout"],
                )
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_1.state_dict().items()
            }
            self.cluster_encoder_1.load_state_dict(state_dict_saved)
            # third
            self.cluster_encoder_2 = conv.GINConv(
                MLP(
                    config["ClusterEncoderChannels"][2],
                    plain_last=False,
                    dropout=config["dropout"],
                )
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_2.state_dict().items()
            }
            self.cluster_encoder_2.load_state_dict(state_dict_saved)
            # linear
            self.linear = Linear(
                config["ClusterEncoderChannels"][-1][-1], config["OutputChannels"]
            )
            state_dict_saved = cluster_net_hetero.linear.state_dict()
            self.linear.load_state_dict(state_dict_saved)

        elif config["conv_type"] == "transformer":
            # first
            self.cluster_encoder_0 = conv.TransformerConv(
                -1,
                out_channels=config["ClusterEncoderOutChannels"][0],
                heads=config["heads"],
                concat=config["concat"],
                beta=config["beta"],
                dropout=config["dropout"],
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_0.state_dict().items()
            }
            self.cluster_encoder_0.load_state_dict(state_dict_saved)
            # second
            self.cluster_encoder_1 = conv.TransformerConv(
                -1,
                out_channels=config["ClusterEncoderOutChannels"][1],
                heads=config["heads"],
                concat=config["concat"],
                beta=config["beta"],
                dropout=config["dropout"],
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_1.state_dict().items()
            }
            self.cluster_encoder_1.load_state_dict(state_dict_saved)
            # third
            self.cluster_encoder_2 = conv.TransformerConv(
                -1,
                out_channels=config["ClusterEncoderOutChannels"][2],
                heads=config["heads"],
                concat=config["concat"],
                beta=config["beta"],
                dropout=config["dropout"],
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_2.state_dict().items()
            }
            self.cluster_encoder_2.load_state_dict(state_dict_saved)
            # linear
            self.linear = Linear(
                config["ClusterEncoderOutChannels"][2], config["OutputChannels"]
            )
            state_dict_saved = cluster_net_hetero.linear.state_dict()
            self.linear.load_state_dict(state_dict_saved)

        else:
            raise ValueError("conv_type should be transformer or ginconv")

    def forward(self, x, edge_index, batch):
        """The method called when ClusterNetHomogeneous is used on a dataitem

        Args:
            x (torch.tensor): cluster features
            edge_index (torch.tensor): contains the edge connections
                between the clusters
            batch (torch.tensor): batch for the clusters

        Returns:
            self.linear(x): Log probabilities for the classes for the
                FOV
        """

        x = self.cluster_encoder_0(x, edge_index)  # .relu()
        x = self.cluster_encoder_1(x, edge_index)  # .relu()
        x = self.cluster_encoder_2(x, edge_index)  # .relu()

        # pooling step so end up with one feature vector per fov
        x = global_max_pool(x, batch)

        # linear layer on each fov feature vector
        return self.linear(x)


class ClusterNetHetero(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "cluster_net"
        if config["conv_type"] == "gin":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="gin",
                    channel_list=config["ClusterEncoderChannels"][0],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="gin",
                    channel_list=config["ClusterEncoderChannels"][1],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="gin",
                    channel_list=config["ClusterEncoderChannels"][2],
                ),
                Linear(
                    config["ClusterEncoderChannels"][-1][-1], config["OutputChannels"]
                ),
            )
        elif config["conv_type"] == "transformer":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    out_channels=config["ClusterEncoderOutChannels"][0],
                    heads=config["heads"],
                    concat=config["concat"],
                    beta=config["beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    out_channels=config["ClusterEncoderOutChannels"][1],
                    heads=config["heads"],
                    concat=config["concat"],
                    beta=config["beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    out_channels=config["ClusterEncoderOutChannels"][2],
                    heads=config["heads"],
                    concat=config["concat"],
                    beta=config["beta"],
                ),
                Linear(
                    config["ClusterEncoderOutChannels"][2], config["OutputChannels"]
                ),
            )
        else:
            raise ValueError("conv_type should be gin or transformer")

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        output = self.cluster_net(x_dict, edge_index_dict, data["clusters"].batch)

        return output.log_softmax(dim=-1)


class ClusterMLP(torch.nn.Module):
    """Simple neural network with series of MLPs

    Attributes:
        name (str): Name of the model
        MLP (nn.Module): MLP for the module
    """

    def __init__(self, config):
        super().__init__()
        self.name = "clustermlp"
        channels = config["channels"]
        self.MLP_in = MLP(
            config["channels"][:-1], plain_last=False, dropout=config["dropout"]
        )
        self.linear = Linear(config["channels"][-2], config["channels"][-1])

    def forward(self, data):
        """Method called when data runs through network

        Args:
            data (torch_geometric.data): Data item that runs through the network

        Raises:
            KeyError: If clusters don't have features

        Returns:
            output.log_softmax(dim=-1): Log probabilities for the classes"""

        # embed each localisation
        try:
            x = data.x_dict["clusters"]
        except KeyError:
            raise KeyError("Clusters need to have features present")
        x = self.MLP_in(x, batch=data["clusters"].batch)
        x = global_mean_pool(x, batch=data["clusters"].batch)
        x = self.linear(x)

        return x.log_softmax(dim=-1)


class LocNet(torch.nn.Module):
    """Neural network that acts on localisations and aggregates into each cluster

    Attributes:
    """

    def __init__(self, config, transformer=False):
        super().__init__()

        self.transformer = transformer
        if not transformer:
            self.name = "locpointnet"
            self.pointnet = PointNetEmbedding(config)
        else:
            self.name = "locpointtransformer"
            self.pointtransformer = PointTransformerEmbedding(config)

    def forward(self, x_dict, pos_dict, edge_index_dict):
        """Method called when data runs through network

        Args:
            x_dict (dict): Features of the locs/clusters
            pos_dict (dict): Positions of the locs/clusters
            edge_index_dict (dict): Edge connections between
                locs/clusters

        Returns:
            output.log_softmax(dim=-1): Log probabilities for the classes"""

        # parse the data

        # get x/pos for locs
        x_locs = x_dict["locs"]
        pos_locs = pos_dict["locs"]

        # get clusterID for each localisation
        clusterID = edge_index_dict["locs", "in", "clusters"][1, :]

        # embed each localisation and aggregate into each cluster
        if not self.transformer:
            x_cluster = self.pointnet(
                x_locs,
                pos_locs,
                batch=clusterID,
            )
        else:
            x_cluster = self.pointtransformer(
                x_locs,
                pos_locs,
                batch=clusterID,
            )
        return x_cluster


class LocNetClassifyFOV(torch.nn.Module):
    """Network that embeds the localisations and makes a prediction for each cluster
    then aggregates the predictions over the FOV to classify the FOV

    Args:
        config (dict): Dictionary containing the configuration for the network
        device (torch.device): Whether to run on cpu or gpu
        transformer (bool): If true use PointTransformer to encode localisations"""

    def __init__(self, config, device="cpu", transformer=False):
        super().__init__()
        self.name = "locnetclassifyfov"
        self.loc_net = LocNet(config, transformer=transformer)
        self.device = device

    def forward(self, data):
        """Method called when data runs through network

        Args:
            data (torch_geometric.data): Data item that runs through the network

        Returns:
            output.log_softmax(dim=-1): Log probabilities for the FOV"""

        # parse data
        x_dict, pos_dict, edge_index_dict, _ = parse_data(data, self.device)

        # get batch ID for each cluster
        cluster_batch = data["clusters"].batch

        # embed each localisation
        x_cluster = self.loc_net(x_dict, pos_dict, edge_index_dict)

        # aggregate over the FOV
        x_fov = global_mean_pool(x_cluster, cluster_batch)

        # return log probability
        return x_fov.log_softmax(dim=-1)


class LocClusterNet(torch.nn.Module):
    """Network that embeds the localisations aggregates this with cluster
    features if present, then embeds the clusters using a graph network before
    using a linear layer to make a prediction for the FOV

    Args:
        config (dict): Dictionary containing the configuration for the network
        device (torch.device): Whether to run on cpu or gpu
        transformer (bool): If true use PointTransformer to encode localisations"""

    def __init__(self, config, device="cpu", transformer=False):
        super().__init__()
        self.name = "locclusternet"
        self.device = device
        # wrong input channel size 2 might change if locs have features
        if not transformer:
            self.loc_net = LocNet(config, transformer=False)
        else:
            self.loc_net = LocNet(config, transformer=True)

        if config["conv_type"] == "gin":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="gin",
                    channel_list=config["ClusterEncoderChannels"][0],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="gin",
                    channel_list=config["ClusterEncoderChannels"][1],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="gin",
                    channel_list=config["ClusterEncoderChannels"][2],
                ),
                Linear(
                    config["ClusterEncoderChannels"][-1][-1], config["OutputChannels"]
                ),
            )
        elif config["conv_type"] == "transformer":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    out_channels=config["ClusterEncoderOutChannels"][0],
                    heads=config["heads"],
                    concat=config["concat"],
                    beta=config["beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    out_channels=config["ClusterEncoderOutChannels"][1],
                    heads=config["heads"],
                    concat=config["concat"],
                    beta=config["beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    out_channels=config["ClusterEncoderOutChannels"][2],
                    heads=config["heads"],
                    concat=config["concat"],
                    beta=config["beta"],
                ),
                Linear(
                    config["ClusterEncoderOutChannels"][2], config["OutputChannels"]
                ),
            )
        else:
            raise ValueError("conv_type should be gin or transformer")

    def forward(self, data):
        """Method called when data runs through network

        Args:
            data (torch_geometric.data): Data item that runs through the network

        Returns:
            output.log_softmax(dim=-1): Log probabilities for the classes"""

        # parse data
        x_dict, pos_dict, edge_index_dict, cluster_feats_present = parse_data(
            data, self.device
        )

        # embed each localisation
        x_cluster = self.loc_net(x_dict, pos_dict, edge_index_dict)

        # apply activation function to cluster embedding to constrain between 0 and 1
        x_cluster = x_cluster.sigmoid()

        # add on cluster features if present
        if cluster_feats_present:
            x_dict["clusters"] = torch.cat((x_dict["clusters"], x_cluster), dim=-1)
        else:
            x_dict["clusters"] = x_cluster

        # operate graph net on clusters, finish with
        output = self.cluster_net(x_dict, edge_index_dict, data["clusters"].batch)

        return output.log_softmax(dim=-1)


# -----------------------------------------------------------------------

# class LocEncoder(torch.nn.Module):
#     """Module that encodes the localisations

#     Attributes:
#         local_nn (torch.nn.Module) : Neural network used by the PointNetConvolution local
#         global_nn (torch.nn.Module) : Neural network used by the PointNetConvolution global
#     """

#     def __init__(self, local_nn, global_nn):
#         super().__init__()
#         # raise ValueError("Need to define how many channels custom")
#         self.conv = PointNetConv(
#             local_nn=local_nn, global_nn=global_nn, add_self_loops=False
#         )

#     def forward(self, x_locs, pos_locs, edge_index_dict):
#         """The method called when the encoder is used on a data item

#         Args:
#             x_locs (torch.tensor): Features of the localisation
#             pos_locs (torch.tensor): Positions of the localisation
#             edge_index_dict (torch.tensor): Edge connections between
#                 localisations

#         Returns:
#             loc_x (torch.tensor): x for the localisations
#         """
#         # raise ValueError("check + do we need a relu?")
#         loc_x = self.conv(
#             (x_locs, x_locs), (pos_locs, pos_locs), edge_index_dict["locs", "clusteredwith", "locs"]
#         )  # .relu()
#         return loc_x


# class LocEncoderTransformer(torch.nn.Module):
#     """Module that encodes the localisations using a PointTransformer

#     Attributes:

#     """


#     def __init__(self, channel_list):
#         super().__init__()
#         # raise ValueError("Need to define how many channels custom")
#         self.transform = TransformerBlock(*channel_list)

#     def forward(self, x_locs, pos_locs, edge_index_dict):
#         """The method called when the encoder is used on a data item

#         Args:
#             x_locs (torch.tensor): Features of the localisation
#             pos_locs (torch.tensor): Positions of the localisation
#             edge_index_dict (torch.tensor): Edge connections between
#                 localisations

#         Returns:
#             x_locs (torch.tensor): x for the localisations
#         """

#         if x_locs is None:
#             x_locs = torch.ones((pos_locs.shape[0], 1), device=pos_locs.get_device())

#         x_locs = self.transform(
#             x_locs, pos_locs, edge_index_dict["locs", "clusteredwith", "locs"]
#         )

#         return x_locs


# class LocNetOnly(torch.nn.Module):
#     """Neural network that acts on localisations but no clusternetwork after

#     Attributes:
#     """

#     def __init__(self, config, device="cpu", transformer=False):
#         super().__init__()
#         self.name = "locnetonly"

#         # wrong input channel size 2 might change if locs have features
#         if not transformer:
#             self.loc_net = LocNet(
#                 encoder_0=LocEncoder(
#                     MLP(
#                         config["LocEncoderChannels_local"][0],
#                         dropout=config["dropout"],
#                         plain_last=False,
#                     ),
#                     MLP(
#                         config["LocEncoderChannels_global"][0],
#                         dropout=config["dropout"],
#                         plain_last=False,
#                     ),
#                 ),
#                 encoder_1=LocEncoder(
#                     MLP(
#                         config["LocEncoderChannels_local"][1],
#                         dropout=config["dropout"],
#                         plain_last=False,
#                     ),
#                     MLP(
#                         config["LocEncoderChannels_global"][1],
#                         dropout=config["dropout"],
#                         plain_last=False,
#                     ),
#                 ),
#                 encoder_2=LocEncoder(
#                     MLP(
#                         config["LocEncoderChannels_local"][2],
#                         dropout=config["dropout"],
#                         plain_last=False,
#                     ),
#                     MLP(
#                         config["LocEncoderChannels_global"][2],
#                         dropout=config["dropout"],
#                         plain_last=False,
#                     ),
#                 ),
#                 loc2cluster=Loc2Cluster(),
#                 device=device,
#             )
#         else:
#             self.loc_net = LocNet(
#                 encoder_0=LocEncoderTransformer(config["LocEncoderTransformer"][0]),
#                 encoder_1=LocEncoderTransformer(config["LocEncoderTransformer"][1]),
#                 encoder_2=LocEncoderTransformer(config["LocEncoderTransformer"][2]),
#                 loc2cluster=Loc2Cluster(),
#                 device=device,
#             )

#     def forward(self, data):
#         """Method called when data runs through network

#         Args:
#             data (torch_geometric.data): Data item that runs through the network

#         Returns:
#             output.log_softmax(dim=-1): Log probabilities for the classes"""
#         # embed each localisation
#         x_dict, _, edge_index_dict = self.loc_net(data)


#         # aggregate over fov
#         x = global_mean_pool(x_dict["clusters"], batch=data["clusters"].batch)

#         # print('output', x.log_softmax(dim=-1).argmax(dim=1))

#         return x.log_softmax(dim=-1)


# class LocNetOld(torch.nn.Module):
#     """Network that encodes the localisations and aggregates to the clusters

#     Attributes:
#         encoder_0 (torch.nn.module): First encoder for the localisations
#         encoder_1 (torch.nn.module): Second encoder for the localisations
#         encoder_2 (torch.nn.module): Third encoder for the localisations
#         loc2cluster (torch.nn.module): Module that aggregates features from localisations
#             to clusters
#         device (torch.device): cpu or gpu"""

#     def __init__(
#         self,
#         encoder_0=None,
#         encoder_1=None,
#         encoder_2=None,
#         loc2cluster=None,
#         device="cpu",
#     ):
#         super().__init__()
#         self.device = device
#         self.encoder_0 = encoder_0
#         self.encoder_1 = encoder_1
#         self.encoder_2 = encoder_2
#         self.loc2cluster = loc2cluster

#     def forward(self, data):
#         """The method called when locnet is used on a dataitem

#         Args:
#             data (torch_geometric.data): Date item from torch geometric
#                 that is passing through the network

#         Returns:
#             x_dict (dict): Features of the localisation
#             pos_dict (dict): Positions of the localisation
#             edge_index_dict (dict): Edge connections between locs/clusters
#         """

#         x_dict, pos_dict, edge_index_dict, cluster_feats_present = parse_data(
#             data, self.device
#         )

#         x_dict["locs"] = self.encoder_0(
#             x_dict["locs"], pos_dict["locs"], edge_index_dict
#         )
#         if self.encoder_1 is not None:
#             x_dict["locs"] = self.encoder_1(
#                 x_dict["locs"], pos_dict["locs"], edge_index_dict
#             )
#         if self.encoder_2 is not None:
#             x_dict["locs"] = self.encoder_2(
#                 x_dict["locs"], pos_dict["locs"], edge_index_dict
#             )
#         # pool the embedding for each localisation to its cluster and concatenate this embedding with previous cluster embedding
#         x_dict["clusters"] = self.loc2cluster(
#             x_dict, edge_index_dict, cluster_feats_present
#         )
#         return x_dict, pos_dict, edge_index_dict

# class Loc2Cluster(torch.nn.Module):
#     """Module that takes the features from localisations and
#     aggregates them to the cluster"""

#     def __init__(self):
#         super().__init__()
#         # raise ValueError("Max or sum?")
#         self.conv = HeteroConv(
#             {("locs", "in", "clusters"): conv.SimpleConv(aggr="max")}, aggr=None
#         )

#     def forward(self, x_dict, edge_index_dict, cluster_feats_present=True):
#         """The method called when this module is used on a data item

#         Args:
#             x_dict (dict): Feature dictionaries, with keys
#                 'clusters' and 'locs' both containing the features
#                 for the respective nodes
#             edge_index_dict (dict): Edge index dictionaries, with keys
#                 for the connections between 'locs'-'locs', 'locs'-'clusters' &
#                 'clusters'-'clusters'
#             cluster_feats_present (bool): Whether the clusters have features
#                 or not

#         Returns:
#             x_dict['clusters'] (torch.tensor): Features for the cluster
#         """
#         #out = self.conv(x_dict, edge_index_dict)
#         # raise ValueError("Do I need torch.squeeze")
#         # raise ValueError("is dimension concatenating in correct")
#         #out["clusters"] = torch.squeeze(out["clusters"])
#         #print(x_dict['locs'])
#         #print(edge_index_dict['locs','in','clusters'].shape)
#         #print(edge_index_dict['locs','in','clusters'])
#         #input('stop')
#         x_clusters = global_max_pool(x_dict['locs'], edge_index_dict['locs','in','clusters'][1,:])
#         if cluster_feats_present:
#             x_dict["clusters"] = torch.cat(
#                 (x_dict["clusters"], x_clusters), dim=-1
#             )
#         else:
#             x_dict["clusters"] = x_clusters
#         return x_dict["clusters"]
