"""This module defines custom transforms to apply to the data"""

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, LinearTransformation
import math
import numbers
import random
from typing import Tuple, Union, Sequence
from itertools import repeat

import torch

# from torch_geometric.nn import (
#    radius,
# )
# from torch_geometric.utils import subgraph
import numpy as np

# import torch
import warnings


# had to change base code as basetransform not implemented yet for me
@functional_transform("subsample")
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

        data_min_x = data.pos[:, 0] > data.pos[idx[0]][0] - self.x / 2
        data_max_x = data.pos[:, 0] < data.pos[idx[0]][0] + self.x / 2
        data_min_y = data.pos[:, 1] > data.pos[idx[0]][1] - self.y / 2
        data_max_y = data.pos[:, 1] < data.pos[idx[0]][1] + self.y / 2

        indices = data_min_x & data_max_x & data_min_y & data_max_y

        warnings.warn("Not implemented for heterogeneous")

        if data.edge_index is not None:
            raise ValueError("Not sure if below works")
        if data.edge_attr is not None:
            raise ValueError("Not implemented")
        if data.edge_index is None and data.edge_attr is None:
            data.x = data.x[indices]
            data.pos = data.pos[indices]
            data.y = data.y[indices]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x: {self.x} y: {self.y})"

@functional_transform('random_rotate_loccluster')
class RandomRotate(BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval. Also rotates cluster locations simulateneously.
    (functional name: :obj:`random_rotate`).

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """
    def __init__(self, degrees: Union[Tuple[float, float], float],
                 axis: int = 0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def forward(self, 
                data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        # check positions for all and all same size
        for index, store in enumerate(data.node_stores):
            if not hasattr(store, 'pos'):
                raise ValueError('No position coordinates')
            else: 
                if index == 0:
                    size = store.pos.size(-1)
                else:
                    assert store.pos.size(-1) == size

        if size == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')
    
@functional_transform('random_jitter_loccluster')
class RandomJitter(BaseTransform):
    r"""Translates node positions by randomly sampled translation values
    within a given interval (functional name: :obj:`random_jitter`).
    In contrast to other random transformations,
    translation is applied separately at each position

    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """
    def __init__(self, translate: Union[float, int, Sequence]):
        self.translate = translate

    def forward(self, 
                data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                (n, dim), t = store.pos.size(), self.translate
                if isinstance(t, numbers.Number):
                    t = list(repeat(t, times=dim))
                assert len(t) == dim

                ts = []
                for d in range(dim):
                    ts.append(store.pos.new_empty(n).uniform_(-abs(t[d]), abs(t[d])))

                store.pos = store.pos + torch.stack(ts, dim=-1)
        
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.translate})'
    
@functional_transform('random_flip_loccluster')
class RandomFlip(BaseTransform):
    """Flips node positions along a given axis randomly with a given
    probability (functional name: :obj:`random_flip`).

    Args:
        axis (int): The axis along the position of nodes being flipped.
        p (float, optional): Probability that node positions will be flipped.
            (default: :obj:`0.5`)
    """
    def __init__(self, axis: int, p: float = 0.5):
        self.axis = axis
        self.p = p

    def forward(self, 
                data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        
        if random.random() < self.p:

            for store in data.node_stores:
                if hasattr(store, 'pos'):
                    pos = store.pos.clone()
                    pos[..., self.axis] = -pos[..., self.axis]
                    store.pos = pos
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(axis={self.axis}, p={self.p})'
    
@functional_transform('random_scale_loccluster')
class RandomScale(BaseTransform):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    (functional name: :obj:`random_scale`)

    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """
    def __init__(self, scales: Tuple[float, float]):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def forward(self, 
                data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        
        scale = random.uniform(*self.scales)

        for store in data.node_stores:
            store.pos = store.pos * scale
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scales})'
    
@functional_transform('random_shear_loccluster')
class RandomShear(BaseTransform):
    r"""Shears node positions by randomly sampled factors :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    (functional name: :obj:`random_shear`)

    .. math::
        \begin{bmatrix}
            1      & s_{xy} & s_{xz} \\
            s_{yx} & 1      & s_{yz} \\
            s_{zx} & z_{zy} & 1      \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        shear (float or int): maximum shearing factor defining the range
            :math:`(-\mathrm{shear}, +\mathrm{shear})` to sample from.
    """
    def __init__(self, shear: Union[float, int]):
        self.shear = abs(shear)

    def forward(self, 
                data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        
         # check positions for all and all same size
        for index, store in enumerate(data.node_stores):
            if not hasattr(store, 'pos'):
                raise ValueError('No position coordinates')
            else: 
                if index == 0:
                    dim = store.pos.size(-1)
                else:
                    assert store.pos.size(-1) == dim
        
        matrix = 2 * torch.rand(dim,dim) - 1
        eye = torch.arange(dim, dtype=torch.long)
        matrix[eye, eye] = 1
        
        return LinearTransformation(matrix)(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.shear})'
