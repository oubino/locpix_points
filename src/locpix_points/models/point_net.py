"""SimplePointNet

PointNets are adapted from
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
and
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py

Originally in
PointNet https://arxiv.org/abs/1612.00593
and
PointNet++ https://arxiv.org/abs/1706.02413
"""

import torch
from torch_geometric.nn import (
    MLP,
    PointNetConv,
    fps,
    global_max_pool,
    knn_interpolate,
    radius,
)
import warnings

# TODO: layer sizes
# make sure sizes multiples of 8 to map onto tensor cores

# note the following is taken directly from example on pytorch geometric
# github


class SAModule(torch.nn.Module):
    """SA module adapted from https://github.com/pyg-team/pytorch_geometric/blob/
    master/examples/pointnet2_classification.py from PointNet/PointNet++

    Args:
        conv (neural net) : PointNetConv from Pytorch Geometric - this
            is the PointNet architecture defining function applied to each
            point"""

    def __init__(self, local_nn, global_nn):
        super().__init__()
        self.conv = PointNetConv(local_nn, global_nn, add_self_loops=False)

    def forward(self, x, pos, edge_index):
        x = self.conv(x, pos, edge_index)
        return x


class GlobalSAModule(torch.nn.Module):
    """Global SA module adapted from https://github.com/pyg-team/pytorch_geometric/blob/
    master/examples/pointnet2_classification.py from PointNet/PointNet++

    Args:
        nn (neural net) : Neural network which acts on points"""

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, clusterID):
        x = self.nn(torch.cat([x, pos], dim=1))
        # aggregate the features for each cluster
        x = global_max_pool(x, clusterID)
        # this is only relevant to segmentation
        pos = pos.new_zeros((x.size(0), pos.shape[-1]))
        # this only works if ClusterID was ordered in the first place
        # as otherwise won't match up
        sorted_clusterID, _ = torch.sort(clusterID)
        assert torch.equal(clusterID, sorted_clusterID)
        clusterID = torch.arange(x.size(0), device=clusterID.device)
        return x, pos, clusterID


class PointNetEmbedding(torch.nn.Module):
    """PointNet embedding model as noted modified from above

    Args:
        config (dictionary) : Configure with these parameters with following keys
        ratio (list) : Ratio of points to sample for each layer
        radius (list) : Radius of neighbourhood to consider for each layer
        channels (list) : Channel sizes for each layer
        dropout (float) : Dropout for the final layer
    """

    def __init__(self, config):
        super().__init__()
        self.name = "PointNetEmbedding"

        local_channels = config["local_channels"]
        global_channels = config["global_channels"]
        global_sa_channels = config["global_sa_channels"]
        final_channels = config["final_channels"]
        dropout = config["dropout"]

        # Input channels account for both `pos` and node features.
        # Note that plain last layers causes issues!!
        self.sa1_module = SAModule(
            MLP(local_channels[0], dropout=dropout, plain_last=False),
            MLP(global_channels[0], dropout=dropout, plain_last=False),
        )
        self.sa2_module = SAModule(
            MLP(local_channels[1], dropout=dropout, plain_last=False),
            MLP(global_channels[1], dropout=dropout, plain_last=False),
        )
        self.sa3_module = GlobalSAModule(
            MLP(global_sa_channels, dropout=dropout, plain_last=False)
        )

        # don't worry, has a plain last layer where no non linearity, norm or dropout
        self.mlp = MLP(final_channels, dropout=dropout, plain_last=True)

        warnings.warn("PointNet embedding requires the clusterID to be ordered")

    def forward(self, x, pos, clusterID, edge_index):
        x = self.sa1_module(x, pos, edge_index)
        x = self.sa2_module(x, pos, edge_index)
        sa3_out = self.sa3_module(x, pos, clusterID)
        x, pos, clusterID = sa3_out

        return self.mlp(x)


class FPModule(torch.nn.Module):
    """FP module adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/
    examples/pointnet2_segmentation.py from PointNet/PointNet++

    Args:
        k (int) :  Number of neighbours to consider during interpolation of features
        nn (neural net) : Net which acts on features for each point"""

    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNetSegmentation(torch.nn.Module):
    """PointNet segmentation model as noted modified from above

    Args:
        config (dictionary) : Configure with these parameters with following keys
            ratio (list) : Ratio of points to sample for each layer
            radius (list) : Radius of neighbourhood to consider for each layer
            sa_channels (list) : Channel sizes for each layer of sa modules
            fp_channels (list) : Channel sizes for each layer of fp modules
            output_channels (list) : Channel sizes for each layer of final MLP
            k (list) : k nearest neighbours to consider for each fp module
            dropout (float) : Dropout for the final layer
    """

    def __init__(self, config):
        super().__init__()
        self.name = "PointNetSegmentation"
        ratio = config["ratio"]
        radius = config["radius"]
        sa_channels = config["sa_channels"]
        fp_channels = config["fp_channels"]
        output_channels = config["output_channels"]
        k = config["k"]
        dropout = config["dropout"]

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(ratio[0], radius[0], MLP(sa_channels[0]))
        self.sa2_module = SAModule(ratio[1], radius[1], MLP(sa_channels[1]))
        self.sa3_module = GlobalSAModule(MLP(sa_channels[2]))

        self.fp3_module = FPModule(k[0], MLP(fp_channels[0]))
        self.fp2_module = FPModule(k[1], MLP(fp_channels[1]))
        self.fp1_module = FPModule(k[2], MLP(fp_channels[2]))

        # don't worry, has a plain last layer where no non linearity, norm or dropout
        self.mlp = MLP(output_channels, dropout=dropout)

        warnings.warn("There are redundant linear layers here...")

        # self.lin1 = torch.nn.Linear(128, 128)
        # self.lin2 = torch.nn.Linear(128, 128)
        # self.lin3 = torch.nn.Linear(128, None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x).log_softmax(dim=-1)
