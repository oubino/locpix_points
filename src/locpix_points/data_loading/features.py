""" Features module

This module defines some feature representations that are used repeatedly
"""

import torch


def load_pos_feat(arrow_table, data, pos, feat, dimensions):
    """Decided how to load in position and features to each node

    Args:
        arrow_table (parquet arrow table) : Data
            in form of parquet file
        data (torch_geometric data) : Data item to
            load position and features into
        pos (string) : How to load in position data
        feat (string) : How to load in feature data
    Returns:
        data (torch_geometric data) : Data with
            position and feature for eacch node"""

    if pos == "xy":
        coord_data, data = xy_pos(arrow_table, data)
    elif pos == "xyz":
        coord_data, data = xyz_pos(arrow_table, data)
    # define coord data in else if position doesn't use

    if feat is None:
        data.x = None
    elif feat == "uniform":
        data.x = torch.ones((data.pos.shape[0],1))
    elif feat == "xy" or feat == "xyz":
        data.x = coord_data

    return data


def xy_pos(arrow_table, data):
    """Load in xy data to each node as position

    Args:
        arrow_table (parquet arrow table) : Data
                in form of parquet file
        data (torch_geometric data) : Data item to load
            position to

    Returns:
        coord_data (tensor) : the coordinates for the
            data point
        data (torch geometric data) : Data item with
            position loaded in now"""

    # convert to tensor (Number of points x 2 (dimensions))
    x = torch.tensor(arrow_table["x"].to_numpy())
    y = torch.tensor(arrow_table["y"].to_numpy())
    coord_data = torch.stack((x, y), dim=1)
    data.pos = coord_data

    return coord_data, data


def xyz_pos(arrow_table, data, dimensions):
    """Load in xyz data to each node as position

    Args:
        arrow_table (parquet arrow table) : Data
                    in form of parquet file
        data (torch_geometric data) : Data item to load
            position to
        dimensions (int) : Dimensions of the data
    Returns:
        coord_data (tensor) : the coordinates for the
            data point
        data (torch geometric data) : Data item with
            position loaded in now"""

    if dimensions == 2:
        raise ValueError("Can't load in xyz data as only have two dimensions")

    # convert to tensor (Number of points x 3 (dimensions))
    x = torch.tensor(arrow_table["x"].to_numpy())
    y = torch.tensor(arrow_table["y"].to_numpy())
    z = torch.tensor(arrow_table["z"].to_numpy())
    coord_data = torch.stack((x, y, z), dim=1)
    data.pos = coord_data

    return coord_data, data
