"""This module defines custom transforms to apply to the data"""

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.nn import (
    radius,
)
from torch_geometric.utils import subgraph
import numpy as np
import torch
import sys

# had to change base code as basetransform not implemented yet for me 
@functional_transform('subsample')
class Subsample(BaseTransform):
    r"""Samples points and features from a point cloud within a circle
    (functional name: :obj:`subsample`).

    Args:
        radius (float): The size of the circle to sample from in nm
    """
    def __init__(
        self,
        radius: float,
    ):
        self.radius = radius

    def forward(self, data: Data) -> Data:

        print('number of data nodes', data.num_nodes)
        
        idx = np.random.choice(data.num_nodes, 1)
        pos = data.pos
        batch = data.batch  
        print('pos shape', pos.shape)
        batch = torch.zeros(data.num_nodes)
        print('batch', data.batch)
        print('pos', data.pos)
        print('radius', self.radius)
        row, col = radius(
            pos, pos[idx], self.radius, batch, batch[idx]
        )
        sys.stdout.flush()
        print('edge index')
        edge_index = torch.stack([col, row], dim=0)

        print(data.edge_index)
        sys.stdout.flush()
        data.edge_index, data.edge_attr = subgraph(col, edge_index, data.edge_attr)  

        print('need to remove isolated nodes')

        transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
        data = transform(data)

        print('number of data nodes', data.num_nodes)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.radius})'