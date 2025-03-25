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
)
from torch_geometric.utils import sort_edge_index, coalesce, contains_self_loops
from torch import Tensor
import torch_geometric.typing
from typing import Optional
from torch_geometric.typing import OptTensor, torch_cluster
import warnings

# TODO: layer sizes
# make sure sizes multiples of 8 to map onto tensor cores

# note the following is taken directly from example on pytorch geometric
# github


# modified from pyg
def radius(
    x,
    y,
    r,
    batch_x=None,
    batch_y=None,
    max_num_neighbors=32,
    num_workers=1,
    batch_size=None,
    ignore_same_index=False,
):
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    .. code-block:: python

        import torch
        from torch_geometric.nn import radius

        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (torch.Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
        ignore_same_index: Foo

    Returns:
        torch_cluster.radius: Radius ..

    .. warning::

        The CPU implementation of :meth:`radius` with :obj:`max_num_neighbors`
        is biased towards certain quadrants.
        Consider setting :obj:`max_num_neighbors` to :obj:`None` or moving
        inputs to GPU before proceeding.
    """
    if not torch_geometric.typing.WITH_TORCH_CLUSTER_BATCH_SIZE:
        return torch_cluster.radius(
            x, y, r, batch_x, batch_y, max_num_neighbors, num_workers, ignore_same_index
        )
    return torch_cluster.radius(
        x,
        y,
        r,
        batch_x,
        batch_y,
        max_num_neighbors,
        num_workers,
        batch_size,
        ignore_same_index,
    )


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
        self.conv = PointNetConv(local_nn, global_nn, add_self_loops=True)

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
        row_old = row
        col_old = col

        row, col = radius(
            # ignore itself
            pos,
            pos,
            self.r,
            batch,
            batch,
            max_num_neighbors=self.k,
            ignore_same_index=True,
        )
        indices = torch.isin(row, idx)
        row = row[indices]
        col = col[indices]
        unique_vals = torch.unique(row)
        mapping = {int(val): id for id, val in enumerate(unique_vals)}
        row = torch.tensor([mapping[int(x)] for x in row], device=row.device)
        print(torch.equal(row, row_old))
        print(torch.equal(col, col_old))
        raise ValueError(
            "bug here as ignore same index is not implemented\
                         yet in the version of pytorch cluster i can install"
        )
        edge_index = torch.stack([col, row], dim=0)
        # remove duplicate edges
        # edge_index = coalesce(edge_index)
        x_dst = None if x is None else x[idx]
        assert not contains_self_loops(
            torch.stack([edge_index[0, :], idx[edge_index[1, :]]])
        )
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
            ratio[0],
            radius[0],
            k,
            MLP(local_channels[0], plain_last=True),  # BN
            MLP(global_channels[0], plain_last=True),  # BN
        )
        self.sa2_module = SAModule(
            ratio[1],
            radius[1],
            k,
            MLP(local_channels[1], plain_last=True),  # BN
            MLP(global_channels[1], plain_last=True),  # BN
        )
        self.sa3_module = SAModule(
            ratio[2],
            radius[2],
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
        x = self.sa1_module(
            x,
            pos,
            batch,
        )
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
