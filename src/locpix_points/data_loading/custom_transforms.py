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
        x (float): The size of the x box in nm
        y (float) : The size of the y box in nm
    """
    def __init__(
        self,
        x: float,
        y: float,
    ):
        self.x = x
        self.y = y

    def forward(self, data: Data) -> Data:

        idx = np.random.choice(data.num_nodes, 1)

        print('number nodes')
        print(data.num_nodes)

        data_min_x = data.pos[:,0] > data.pos[idx[0]][0] - self.x/2
        data_max_x = data.pos[:,0] < data.pos[idx[0]][0] + self.x/2
        data_min_y = data.pos[:,1] > data.pos[idx[0]][1] - self.y/2
        data_max_y = data.pos[:,1] < data.pos[idx[0]][1] + self.y/2

        data.pos = data.pos[data_min_x&data_max_x&data_min_y&data_max_y]
        data.x = data.x[data_min_x&data_max_x&data_min_y&data_max_y]

        print('number nodes')
        print(data.num_nodes)
    
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.radius})'