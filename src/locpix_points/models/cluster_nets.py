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
from torch_geometric.nn import MLP, HeteroConv, conv
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from .point_transformer import PointTransformerEmbedding
from .point_net import PointNetEmbedding


class ClusterEncoder(torch.nn.Module):
    """Module for encoding clusters

    Attributes:
        channel_list (list): Channel sizes for the MLP used in the neural network
            used in Conv
        dropout (int): Dropout to apply to MLP
        conv_type (str): Either gin, transformer, pointnet or pointtransformer
        channel_list (list): If conv_type == "gin" or "pointnet" - Channels for MLP
        tr_out_channels (int): If conv_type == "transformer" - Out channels
        tr_heads (int): If conv_type == "transformer" - Number of multihead attention heads
        tr_concat (bool): If conv_type == "transformer" - If true concatenate features from
            heads, if false then average
        tr_beta (bool): If conv_type == "transformer" - See pytorch geometric for more details
        pt_tr_in_channels (int): If conv_type == "pointtransformer" - Input channels
        pt_tr_out_channels (int): If conv_type == "pointtransformer" - Ouput channels
        pt_tr_pos_nn_layers (int): If conv_type == "pointtransformer" - Hidden channels for
            position neural network
        pt_tr_attn_nn_layers (int): If conv_type == "pointtransformer" - Hidden channels for
            attention neural network
        pt_tr_dim (int): If conv_type == "pointtransformer" - Dimensions for coordinates of points
    """

    def __init__(
        self,
        dropout,
        conv_type="gin",
        channel_list=None,
        tr_out_channels=None,
        tr_heads=1,
        tr_concat=True,
        tr_beta=False,
        pt_tr_in_channels=None,
        pt_tr_out_channels=None,
        pt_tr_pos_nn_layers=None,
        pt_tr_attn_nn_layers=None,
        pt_tr_dim=2,
    ):
        super().__init__()
        self.conv_type = conv_type
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
                        tr_out_channels,
                        heads=tr_heads,
                        concat=tr_concat,
                        beta=tr_beta,
                        dropout=dropout,
                    )
                },
                aggr="max",
            )
        elif conv_type == "pointnet":
            nn = MLP(channel_list, plain_last=False, dropout=dropout)
            self.conv = HeteroConv(
                {
                    ("clusters", "near", "clusters"): conv.PointNetConv(
                        nn, add_self_loops=False
                    )
                },
                aggr="max",
            )
        elif conv_type == "pointtransformer":
            pos_nn = MLP(
                [pt_tr_dim, pt_tr_pos_nn_layers, pt_tr_out_channels],
                plain_last=False,
                dropout=dropout,
            )
            attn_nn = MLP(
                [pt_tr_out_channels, pt_tr_attn_nn_layers, pt_tr_out_channels],
                plain_last=False,
                dropout=dropout,
            )
            self.conv = HeteroConv(
                {
                    ("clusters", "near", "clusters"): conv.PointTransformerConv(
                        pt_tr_in_channels,
                        pt_tr_out_channels,
                        pos_nn,
                        attn_nn,
                        add_self_loops=False,
                    )
                },
                aggr="max",
            )
        else:
            raise ValueError(f"{conv_type} not supported")

    def forward(self, x_dict, pos_dict, edge_index_dict, add_cluster_pos=False):
        """The method called when the encoder is used on a data item

        Args:
            x_dict (dict): Features of the locs/clusters
            pos_dict (dict): Postiions of the locs/clusters
            edge_index_dict (dict): Edge connections between
                locs/clusters
            add_cluster_pos (bool): Add on the cluster positions

        Returns:
            out["clusters"].relu() (torch.tensor): Encoded cluster features
        """
        if add_cluster_pos:
            x_dict["clusters"] = torch.cat(
                (x_dict["clusters"], pos_dict["clusters"]), dim=-1
            )
        if self.conv_type in ["gin", "transformer"]:
            out = self.conv(x_dict, edge_index_dict)
        elif self.conv_type in ["pointnet", "pointtransformer"]:
            out = self.conv(x_dict, pos_dict, edge_index_dict)
        return out["clusters"]
        # raise ValueError("Wrong axis when have batch")


class ClusterNet(torch.nn.Module):
    """Network for taking the cluster embeddings and making a classification

    Attributes:
        cluster_encoder_0 (torch.nn.module): First encoder for the clusters
        cluster_encoder_1 (torch.nn.module): Second encoder for the clusters
        cluster_encoder_2 (torch.nn.module): Third encoder for the clusters
        cluster_encoder_3 (torch.nn.module): Fourth encoder for the clusters
        linear (torch.nn.module): Linear layer that operates on cluster embeddings
            and returns a classification
    """

    def __init__(
        self,
        cluster_encoder_0,
        cluster_encoder_1,
        cluster_encoder_2,
        cluster_encoder_3,
        linear,
    ):
        super().__init__()
        self.cluster_encoder_0 = cluster_encoder_0
        self.cluster_encoder_1 = cluster_encoder_1
        self.cluster_encoder_2 = cluster_encoder_2
        self.cluster_encoder_3 = cluster_encoder_3
        self.linear = linear

    def forward(self, x_dict, pos_dict, edge_index_dict, batch, add_cluster_pos):
        """The method called when ClusterNet is used on a dataitem

        Args:
            x_dict (dict): dictionary with the features for the
                clusters/locs
            pos_dict (dict): dictionary with the positions for the
                clusters/locs
            edge_index_dict (dict): contains the edge connections
                between the locs/clusters
            add_cluster_pos (bool): if True add on position for each
                cluster
            batch (torch.tensor): batch for the clusters

        Returns:
            self.linear(x_dict['clusters']): Log-probability for the classes
                for that FOV
        """
        x_dict["clusters"] = self.cluster_encoder_0(
            x_dict, pos_dict, edge_index_dict, add_cluster_pos=add_cluster_pos
        )
        x_dict["clusters"] = self.cluster_encoder_1(
            x_dict, pos_dict, edge_index_dict, add_cluster_pos=add_cluster_pos
        )
        x_dict["clusters"] = self.cluster_encoder_2(
            x_dict, pos_dict, edge_index_dict, add_cluster_pos=add_cluster_pos
        )
        x_dict["clusters"] = self.cluster_encoder_3(
            x_dict, pos_dict, edge_index_dict, add_cluster_pos=add_cluster_pos
        )

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
        self.name = "ClusterNetHomogeneous"
        self.add_cluster_pos = config["add_cluster_pos"]
        self.conv_type = config["cluster_conv_type"]

        if config["cluster_conv_type"] == "gin":
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
            self.cluster_encoder_0.aggr = "max"
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
            self.cluster_encoder_1.aggr = "max"
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
            self.cluster_encoder_2.aggr = "max"
            # linear
            self.linear = Linear(
                config["ClusterEncoderChannels"][-1][-1], config["OutputChannels"]
            )
            state_dict_saved = cluster_net_hetero.linear.state_dict()
            self.linear.load_state_dict(state_dict_saved)

        elif config["cluster_conv_type"] == "transformer":
            # first
            self.cluster_encoder_0 = conv.TransformerConv(
                -1,
                out_channels=config["tr_out_channels"][0],
                heads=config["tr_heads"],
                concat=config["tr_concat"],
                beta=config["tr_beta"],
                dropout=config["dropout"],
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_0.state_dict().items()
            }
            self.cluster_encoder_0.load_state_dict(state_dict_saved)
            self.cluster_encoder_0.aggr = "max"
            # second
            self.cluster_encoder_1 = conv.TransformerConv(
                -1,
                out_channels=config["tr_out_channels"][1],
                heads=config["tr_heads"],
                concat=config["tr_concat"],
                beta=config["tr_beta"],
                dropout=config["dropout"],
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_1.state_dict().items()
            }
            self.cluster_encoder_1.load_state_dict(state_dict_saved)
            self.cluster_encoder_1.aggr = "max"
            # third
            self.cluster_encoder_2 = conv.TransformerConv(
                -1,
                out_channels=config["tr_out_channels"][2],
                heads=config["tr_heads"],
                concat=config["tr_concat"],
                beta=config["tr_beta"],
                dropout=config["dropout"],
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_2.state_dict().items()
            }
            self.cluster_encoder_2.load_state_dict(state_dict_saved)
            self.cluster_encoder_2.aggr = "max"
            # linear
            self.linear = Linear(config["tr_out_channels"][2], config["OutputChannels"])
            state_dict_saved = cluster_net_hetero.linear.state_dict()
            self.linear.load_state_dict(state_dict_saved)

        elif config["cluster_conv_type"] == "pointnet":
            # first
            self.cluster_encoder_0 = conv.PointNetConv(
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
            self.cluster_encoder_0.aggr = "max"
            # second
            self.cluster_encoder_1 = conv.PointNetConv(
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
            self.cluster_encoder_1.aggr = "max"
            # third
            self.cluster_encoder_2 = conv.PointNetConv(
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
            self.cluster_encoder_2.aggr = "max"
            # linear
            self.linear = Linear(
                config["ClusterEncoderChannels"][-1][-1], config["OutputChannels"]
            )
            state_dict_saved = cluster_net_hetero.linear.state_dict()
            self.linear.load_state_dict(state_dict_saved)

        elif config["cluster_conv_type"] == "pointtransformer":
            # first
            self.cluster_encoder_0 = conv.PointTransformerConv(
                config["pt_tr_in_channels"][0],
                config["pt_tr_out_channels"][0],
                MLP(
                    [
                        config["pt_tr_dim"],
                        config["pt_tr_pos_nn_layers"],
                        config["pt_tr_out_channels"][0],
                    ],
                    plain_last=False,
                    dropout=config["dropout"],
                ),
                MLP(
                    [
                        config["pt_tr_out_channels"][0],
                        config["pt_tr_attn_nn_layers"],
                        config["pt_tr_out_channels"][0],
                    ],
                    plain_last=False,
                    dropout=config["dropout"],
                ),
                add_self_loops=False,
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_0.state_dict().items()
            }
            self.cluster_encoder_0.load_state_dict(state_dict_saved)
            self.cluster_encoder_0.aggr = "max"
            # second
            self.cluster_encoder_1 = conv.PointTransformerConv(
                config["pt_tr_in_channels"][1],
                config["pt_tr_out_channels"][1],
                MLP(
                    [
                        config["pt_tr_dim"],
                        config["pt_tr_pos_nn_layers"],
                        config["pt_tr_out_channels"][1],
                    ],
                    plain_last=False,
                    dropout=config["dropout"],
                ),
                MLP(
                    [
                        config["pt_tr_out_channels"][1],
                        config["pt_tr_attn_nn_layers"],
                        config["pt_tr_out_channels"][1],
                    ],
                    plain_last=False,
                    dropout=config["dropout"],
                ),
                add_self_loops=False,
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_1.state_dict().items()
            }
            self.cluster_encoder_1.load_state_dict(state_dict_saved)
            self.cluster_encoder_1.aggr = "max"
            # third
            self.cluster_encoder_2 = conv.PointTransformerConv(
                config["pt_tr_in_channels"][2],
                config["pt_tr_out_channels"][2],
                MLP(
                    [
                        config["pt_tr_dim"],
                        config["pt_tr_pos_nn_layers"],
                        config["pt_tr_out_channels"][2],
                    ],
                    plain_last=False,
                    dropout=config["dropout"],
                ),
                MLP(
                    [
                        config["pt_tr_out_channels"][2],
                        config["pt_tr_attn_nn_layers"],
                        config["pt_tr_out_channels"][2],
                    ],
                    plain_last=False,
                    dropout=config["dropout"],
                ),
                add_self_loops=False,
            )
            state_dict_saved = {
                key[40:]: value
                for key, value in cluster_net_hetero.cluster_encoder_2.state_dict().items()
            }
            self.cluster_encoder_2.load_state_dict(state_dict_saved)
            self.cluster_encoder_2.aggr = "max"
            # linear
            self.linear = Linear(
                config["pt_tr_out_channels"][2], config["OutputChannels"]
            )
            state_dict_saved = cluster_net_hetero.linear.state_dict()
            self.linear.load_state_dict(state_dict_saved)

        else:
            raise ValueError("conv_type should be transformer or ginconv")

    def forward(self, x, edge_index, batch, pos, logits=True):
        """The method called when ClusterNetHomogeneous is used on a dataitem

        Args:
            x (torch.tensor): cluster features
            edge_index (torch.tensor): contains the edge connections
                between the clusters
            batch (torch.tensor): batch for the clusters
            pos (torch.tensor): Pposition for the clusters
            logits (bool): If true output logits, if false output log probs

        Returns:
            self.linear(x): Log probabilities for the classes for the
                FOV
        """

        if self.add_cluster_pos:
            x = torch.cat((x, pos), dim=-1)
        if self.conv_type in ["gin", "transformer"]:
            x = self.cluster_encoder_0(x, edge_index)
        elif self.conv_type in ["pointnet", "pointtransformer"]:
            x = self.cluster_encoder_0(x, pos, edge_index)
        if self.add_cluster_pos:
            x = torch.cat((x, pos), dim=-1)
        if self.conv_type in ["gin", "transformer"]:
            x = self.cluster_encoder_1(x, edge_index)
        elif self.conv_type in ["pointnet", "pointtransformer"]:
            x = self.cluster_encoder_1(x, pos, edge_index)
        if self.add_cluster_pos:
            x = torch.cat((x, pos), dim=-1)
        if self.conv_type in ["gin", "transformer"]:
            x = self.cluster_encoder_2(x, edge_index)
        elif self.conv_type in ["pointnet", "pointtransformer"]:
            x = self.cluster_encoder_2(x, pos, edge_index)

        # pooling step so end up with one feature vector per fov
        x = global_max_pool(x, batch)

        # linear layer on each fov feature vector
        if logits:
            return self.linear(x)
        else:
            logits = self.linear(x)
            return logits.log_softmax(dim=-1)


class ClusterNetHetero(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "cluster_net"
        self.add_cluster_pos = config["add_cluster_pos"]
        if config["cluster_conv_type"] == "gin":
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
        elif config["cluster_conv_type"] == "transformer":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    tr_out_channels=config["tr_out_channels"][0],
                    tr_heads=config["tr_heads"],
                    tr_concat=config["tr_concat"],
                    tr_beta=config["tr_beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    tr_out_channels=config["tr_out_channels"][1],
                    tr_heads=config["tr_heads"],
                    tr_concat=config["tr_concat"],
                    tr_beta=config["tr_beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    tr_out_channels=config["tr_out_channels"][2],
                    tr_heads=config["tr_heads"],
                    tr_concat=config["tr_concat"],
                    tr_beta=config["tr_beta"],
                ),
                Linear(config["tr_out_channels"][2], config["OutputChannels"]),
            )
        elif config["cluster_conv_type"] == "pointnet":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointnet",
                    channel_list=config["ClusterEncoderChannels"][0],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointnet",
                    channel_list=config["ClusterEncoderChannels"][1],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointnet",
                    channel_list=config["ClusterEncoderChannels"][2],
                ),
                Linear(
                    config["ClusterEncoderChannels"][-1][-1], config["OutputChannels"]
                ),
            )
        elif config["cluster_conv_type"] == "pointtransformer":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointtransformer",
                    pt_tr_in_channels=config["pt_tr_in_channels"][0],
                    pt_tr_out_channels=config["pt_tr_out_channels"][0],
                    pt_tr_pos_nn_layers=config["pt_tr_pos_nn_layers"],
                    pt_tr_attn_nn_layers=config["pt_tr_attn_nn_layers"],
                    pt_tr_dim=config["pt_tr_dim"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointtransformer",
                    pt_tr_in_channels=config["pt_tr_in_channels"][1],
                    pt_tr_out_channels=config["pt_tr_out_channels"][1],
                    pt_tr_pos_nn_layers=config["pt_tr_pos_nn_layers"],
                    pt_tr_attn_nn_layers=config["pt_tr_attn_nn_layers"],
                    pt_tr_dim=config["pt_tr_dim"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointtransformer",
                    pt_tr_in_channels=config["pt_tr_in_channels"][2],
                    pt_tr_out_channels=config["pt_tr_out_channels"][2],
                    pt_tr_pos_nn_layers=config["pt_tr_pos_nn_layers"],
                    pt_tr_attn_nn_layers=config["pt_tr_attn_nn_layers"],
                    pt_tr_dim=config["pt_tr_dim"],
                ),
                Linear(config["pt_tr_out_channels"][2], config["OutputChannels"]),
            )
        else:
            conv_type = config["cluster_conv_type"]
            raise NotImplementedError(f"{conv_type} is not implemented")

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        pos_dict = data.pos_dict

        output = self.cluster_net(
            x_dict,
            pos_dict,
            edge_index_dict,
            data["clusters"].batch,
            self.add_cluster_pos,
        )

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
        x = global_max_pool(x, batch=data["clusters"].batch)
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

    def forward(self, x_locs, edge_index_locs, pos_locs):
        """Method called when data runs through network
        Note the order of the arguments MUST not be changed as is required in this order
        for explainability

        Args:
            x_locs (array): Features of the locs
            edge_index_locs (array): Edge connections between
                locs
            pos_locs (array): Positions of the locs

        Returns:
            output.log_softmax(dim=-1): Log probabilities for the classes"""

        # get clusterID for each localisation
        clusterID = edge_index_locs[1, :]

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
        x_cluster = self.loc_net(
            x_locs=x_dict["locs"],
            edge_index_locs=edge_index_dict["locs", "in", "clusters"],
            pos_locs=pos_dict["locs"],
        )

        # aggregate over the FOV
        x_fov = global_max_pool(x_cluster, cluster_batch)

        # return log probability
        return x_fov.log_softmax(dim=-1)


class LocClusterNet(torch.nn.Module):
    """Network that embeds the localisations aggregates this with cluster
    features if present, then embeds the clusters using a graph network before
    using a linear layer to make a prediction for the FOV

    Args:
        config (dict): Dictionary containing the configuration for the network
        device (torch.device): Whether to run on cpu or gpu

    Raises:
        NotImplementedError: If incorrect loc convolution specified"""

    def __init__(self, config, device="cpu"):
        super().__init__()
        self.name = "locclusternet"
        self.device = device
        self.add_cluster_pos = config["add_cluster_pos"]
        if config["loc_conv_type"] == "pointtransformer":
            transformer = True
        elif config["loc_conv_type"] == "pointnet":
            transformer = False
        else:
            raise NotImplementedError(
                "Loc conv type should be pointnet or pointtransformer"
            )

        # wrong input channel size 2 might change if locs have features
        self.loc_net = LocNet(config, transformer=transformer)

        if config["cluster_conv_type"] == "gin":
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
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="gin",
                    channel_list=config["ClusterEncoderChannels"][3],
                ),
                Linear(
                    config["ClusterEncoderChannels"][-1][-1], config["OutputChannels"]
                ),
            )
        elif config["cluster_conv_type"] == "transformer":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    tr_out_channels=config["tr_out_channels"][0],
                    tr_heads=config["tr_heads"],
                    tr_concat=config["tr_concat"],
                    tr_beta=config["tr_beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    tr_out_channels=config["tr_out_channels"][1],
                    tr_heads=config["tr_heads"],
                    tr_concat=config["tr_concat"],
                    tr_beta=config["tr_beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    tr_out_channels=config["tr_out_channels"][2],
                    tr_heads=config["tr_heads"],
                    tr_concat=config["tr_concat"],
                    tr_beta=config["tr_beta"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="transformer",
                    tr_out_channels=config["tr_out_channels"][3],
                    tr_heads=config["tr_heads"],
                    tr_concat=config["tr_concat"],
                    tr_beta=config["tr_beta"],
                ),
                Linear(config["tr_out_channels"][-1], config["OutputChannels"]),
            )
        elif config["cluster_conv_type"] == "pointnet":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointnet",
                    channel_list=config["ClusterEncoderChannels"][0],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointnet",
                    channel_list=config["ClusterEncoderChannels"][1],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointnet",
                    channel_list=config["ClusterEncoderChannels"][2],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointnet",
                    channel_list=config["ClusterEncoderChannels"][3],
                ),
                Linear(
                    config["ClusterEncoderChannels"][-1][-1], config["OutputChannels"]
                ),
            )
        elif config["cluster_conv_type"] == "pointtransformer":
            self.cluster_net = ClusterNet(
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointtransformer",
                    pt_tr_in_channels=config["pt_tr_in_channels"][0],
                    pt_tr_out_channels=config["pt_tr_out_channels"][0],
                    pt_tr_pos_nn_layers=config["pt_tr_pos_nn_layers"],
                    pt_tr_attn_nn_layers=config["pt_tr_attn_nn_layers"],
                    pt_tr_dim=config["pt_tr_dim"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointtransformer",
                    pt_tr_in_channels=config["pt_tr_in_channels"][1],
                    pt_tr_out_channels=config["pt_tr_out_channels"][1],
                    pt_tr_pos_nn_layers=config["pt_tr_pos_nn_layers"],
                    pt_tr_attn_nn_layers=config["pt_tr_attn_nn_layers"],
                    pt_tr_dim=config["pt_tr_dim"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointtransformer",
                    pt_tr_in_channels=config["pt_tr_in_channels"][2],
                    pt_tr_out_channels=config["pt_tr_out_channels"][2],
                    pt_tr_pos_nn_layers=config["pt_tr_pos_nn_layers"],
                    pt_tr_attn_nn_layers=config["pt_tr_attn_nn_layers"],
                    pt_tr_dim=config["pt_tr_dim"],
                ),
                ClusterEncoder(
                    dropout=config["dropout"],
                    conv_type="pointtransformer",
                    pt_tr_in_channels=config["pt_tr_in_channels"][3],
                    pt_tr_out_channels=config["pt_tr_out_channels"][3],
                    pt_tr_pos_nn_layers=config["pt_tr_pos_nn_layers"],
                    pt_tr_attn_nn_layers=config["pt_tr_attn_nn_layers"],
                    pt_tr_dim=config["pt_tr_dim"],
                ),
                Linear(config["pt_tr_out_channels"][-1], config["OutputChannels"]),
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
        x_cluster = self.loc_net(
            x_locs=x_dict["locs"],
            edge_index_locs=edge_index_dict["locs", "in", "clusters"],
            pos_locs=pos_dict["locs"],
        )

        # apply activation function to cluster embedding to constrain between 0 and 1
        x_cluster = x_cluster.sigmoid()

        # add on cluster features if present
        if cluster_feats_present:
            x_dict["clusters"] = torch.cat((x_dict["clusters"], x_cluster), dim=-1)
        else:
            x_dict["clusters"] = x_cluster

        # operate graph net on clusters
        output = self.cluster_net(
            x_dict,
            pos_dict,
            edge_index_dict,
            data["clusters"].batch,
            self.add_cluster_pos,
        )

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
