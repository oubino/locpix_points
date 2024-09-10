"""Networks that only have localisations and no clusters!"""

import torch
from .point_net import PointNetEmbedding
from .point_transformer import PointTransformerEmbedding


class LocOnlyNet(torch.nn.Module):
    """f

    Args:
        config (dict): Dictionary containing the configuration for the network
        device (torch.device): Whether to run on cpu or gpu

    Raises:
    """

    def __init__(self, config, device="cpu"):
        super().__init__()
        self.name = "loconlynet"
        self.device = device
        if config["conv_type"] == "pointtransformer":
            self.net = PointTransformerEmbedding(config)
        elif config["conv_type"] == "pointnet":
            self.net = PointNetEmbedding(config)
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

        output = self.net(
            data.x,
            data.pos,
            data.batch,
        )

        return output.log_softmax(dim=-1)
