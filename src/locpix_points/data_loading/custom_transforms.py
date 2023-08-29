"""This module defines custom transforms to apply to the data"""

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
#from torch_geometric.nn import (
#    radius,
#)
#from torch_geometric.utils import subgraph
import numpy as np
#import torch
#import sys
import warnings

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

        print('number nodes pre')
        print(data.num_nodes)
        print(data.x)
        print(data.pos)
        print(data.y)

        data_min_x = data.pos[:,0] > data.pos[idx[0]][0] - self.x/2
        data_max_x = data.pos[:,0] < data.pos[idx[0]][0] + self.x/2
        data_min_y = data.pos[:,1] > data.pos[idx[0]][1] - self.y/2
        data_max_y = data.pos[:,1] < data.pos[idx[0]][1] + self.y/2
        
        indices = data_min_x&data_max_x&data_min_y&data_max_y
        
        warnings.warn('Not implemented for heterogeneous')

        if data.edge_index is not None:
            raise ValueError('Not sure if below works')
            data = data.subgraph(indices)
        if data.edge_attr is not None:
            raise ValueError('Not implemented')
        if data.edge_index is None and data.edge_attr is None:
            data.x = data.x[indices]
            data.pos = data.pos[indices]
            data.y = data.y[indices]

        print('number nodes post')
        print(data.num_nodes)
        print(data.x)
        print(data.pos)
        print(data.y)
    
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.radius})'