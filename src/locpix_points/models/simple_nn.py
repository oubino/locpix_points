"""Simple Neural Network

Network takes in cluster features and uses simple MLP without using edges


"""

import torch
from torch_geometric.nn import MLP
from torch_geometric.nn.pool import global_mean_pool


class ClusterMLP(torch.nn.Module):
    """Simple neural network with series of MLPs

    Attributes:
        name (str): Name of the model
        MLP (nn.Module): MLP for the module
    """

    def __init__(self, config):
        super().__init__()
        self.name = "clustermlp"
        channels = config["channels"]
        # needs to have two channels at end for each probability
        # needs to have 7 channels input for the cluster features
        assert channels[0] == 7
        assert channels[-1] == 2
        self.MLP = MLP(config["channels"])

    def forward(self, data):
        """Method called when data runs through network

        Args:
            data (torch_geometric.data): Data item that runs through the network

        Raises:
            KeyError: If clusters don't have features

        Returns:
            output.log_softmax(dim=-1): Log probabilities for the classes"""

        # embed each localisation
        try:
            x = data.x_dict["clusters"]
        except KeyError:
            raise KeyError("Clusters need to have features present")
        x = self.MLP(x, batch=data["clusters"].batch)
        x = global_mean_pool(x, batch=data["clusters"].batch)

        return x.log_softmax(dim=-1)
