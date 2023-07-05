"""SimplePointNet

Test PointNet model very basic
and will be deleted was here for
purpose of code experimenting

PointNets are adapted from 
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
and 
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate

# TODO: layer sizes
# make sure sizes multiples of 8 to map onto tensor cores

# note the following is taken directly from example on pytorch geometric
# github


class SAModule(torch.nn.Module):

    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # ------ fps ---------
        # this generates indices to sample from data
        # first index represents random value from pos
        # all subsequent indices represent values furthest from pos
        # ratio defines how many points to sample
        idx = fps(pos, batch, ratio=self.ratio)
        # ------ radius -------
        # finds for each element in pos[idx] all points in pos
        # within distance self.r
        # row is the pos[idx] indices
        # e.g. [0,0,1,1,2,2] - first, second, third points
        # col is the index of the nearest points to these
        # e.g. [1,0,2,1,3,0]
        # this all means that
        # pos[idx][0] is nearest to pos[1] and pos[0]
        # pos[idx][1] is nearest to pos[2] and pos[1]
        # pos[idx][2] is nearest to pos[3] and pos[0]
        #row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
        #                  max_num_neighbors=64)
        #edge_index = torch.stack([col, row], dim=0)
        edge_index = radius(pos,
                            pos[idx],
                            self.r,
                            batch,
                            batch[idx],
                            max_num_neighbors=64)
        row, col = edge_index
        # don't really get this as i think ends up just being same as if 
        # they hadn't split row and col in first place need to check this!
        new_edge_index = torch.stack([col, row], dim=0)
        if edge_index==new_edge_index:
            input('stop!!!')
        else:
            input('stop and check difference')

        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
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

                                                                                                                                                                               
class PointNetClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)


class PointNetSegmentation(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x).log_softmax(dim=-1)


