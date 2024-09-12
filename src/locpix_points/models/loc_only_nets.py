"""Networks that only have localisations and no clusters!"""

import torch
from .point_net import PointNetEmbedding
from .point_transformer import PointTransformerEmbedding


class LocOnlyNet(torch.nn.Module):
    """f

    Args:
        config (dict): Dictionary containing the configuration for the network
        device (torch.device): Whether to run on cpu or gpu
        static (bool): If true edge indices are pre-calculated

    Raises:
    """

    def __init__(self, config, device="cpu", static=False):
        super().__init__()
        self.name = "loconlynet"
        self.device = device
        self.static = static
        if config["conv_type"] == "pointtransformer":
            self.net = PointTransformerEmbedding(config, static)
        elif config["conv_type"] == "pointnet":
            self.net = PointNetEmbedding(config, static)
        else:
            raise NotImplementedError(
                "Loc conv type should be pointnet or pointtransformer"
            )

    def forward(self, data):
        """Method called when data runs through network

        Args:
            data (torch_geometric.data): Data item that runs through the network

        Returns:
            output.log_softmax(dim=-1): Log probabilities for the classes"""

        if self.static:
            edge_index = data.edge_index
        else:
            edge_index = None

        output = self.net(
            data.x,
            data.pos,
            data.batch,
            edge_index=edge_index,
        )

        return output.log_softmax(dim=-1)
