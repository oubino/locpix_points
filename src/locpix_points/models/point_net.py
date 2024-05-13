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
from torch_geometric.utils import sort_edge_index, coalesce
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

    def __init__(self, ratio, r, k, local_nn, global_nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.k = k
        self.conv = PointNetConv(local_nn, global_nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            # add one to nearest neighs as nearest neighs includes itself
            pos,
            pos[idx],
            self.r,
            batch,
            batch[idx],
            max_num_neighbors=self.k + 1,
        )
        edge_index = torch.stack([col, row], dim=0)
        # remove duplicate edges
        edge_index = coalesce(edge_index)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """Global SA module adapted from https://github.com/pyg-team/pytorch_geometric/blob/
    master/examples/pointnet2_classification.py from PointNet/PointNet++

    Args:
        nn (neural net) : Neural network which acts on points"""

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        # aggregate the features for each cluster
        sorted_batch, _ = torch.sort(batch)
        assert torch.equal(batch, sorted_batch)
        x = global_max_pool(x, batch)
        # this is only relevant to segmentation
        pos = pos.new_zeros((x.size(0), pos.shape[-1]))
        # this only works if batch was ordered in the first place
        # as otherwise won't match up
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


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
        k = config["k"]
        radius = config["radius"]
        ratio = config["ratio"]

        # Input channels account for both `pos` and node features.
        # Note that plain last layers causes issues!!
        self.sa1_module = SAModule(
            ratio,
            radius,
            k,
            MLP(local_channels[0], plain_last=True),  # BN
            MLP(global_channels[0], plain_last=True),  # BN
        )
        self.sa2_module = SAModule(
            ratio,
            radius,
            k,
            MLP(local_channels[1], plain_last=True),  # BN
            MLP(global_channels[1], plain_last=True),  # BN
        )
        self.sa3_module = SAModule(
            ratio,
            radius,
            k,
            MLP(local_channels[2], plain_last=True),  # BN
            MLP(global_channels[2], plain_last=True),  # BN
        )
        self.sa4_module = GlobalSAModule(MLP(global_sa_channels, plain_last=True))  # BN

        # don't worry, has a plain last layer where no non linearity, norm or dropout
        self.mlp = MLP(
            final_channels, dropout=dropout, norm=None, plain_last=True
        )  # BN

        warnings.warn("PointNet embedding requires the clusterID to be ordered")

    def forward(self, x, pos, batch):
        x = self.sa1_module(x, pos, batch)
        x = self.sa2_module(*x)
        x = self.sa3_module(*x)
        sa4_out = self.sa4_module(*x)
        x, pos, batch = sa4_out

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
