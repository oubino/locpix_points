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
        _, col = radius(
            pos, pos[idx], self.radius, batch, batch[idx]
        )
        data.edge_index, data.edge_attr = subgraph(col, data.edge_index, data.edge_attr)  

        print('number of data nodes', data.num_nodes)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.radius})'