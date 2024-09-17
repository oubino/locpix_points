"""Networks that only have localisations and no clusters!"""

import torch
from .point_net import PointNetEmbedding
from .point_transformer import PointTransformerEmbedding
from torch_geometric.nn import global_max_pool


class LocOnlyNet(torch.nn.Module):
    """Neural network that takes in only localisations

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
            self.net = PointTransformerEmbedding(config)
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


class LocOnlyNetExplain(LocOnlyNet):
    """Neural network that takes in only localisations used for the explain  module

    Args:
        config (dict): Dictionary containing the configuration for the network
        device (torch.device): Whether to run on cpu or gpu
        static (bool): If true edge indices are pre-calculated
    """

    def __init__(self, config, device="cpu"):
        super().__init__(config, device=device, static=True)

    def forward(self, x, edge_index, batch, pos, logits=True):
        """Method called when data runs through network

        Args:
            x (torch.tensor): Input feature that runs through the network
            edge_index (torch.tensor): Edge connections for the input graph
            batch (torch.tensor): Batch for the input graph
            pos (torch.tensor): Positions for the input graph
            logits (bool): Whether to output logits

        Returns:
            output (torch.tensor): If logits is True returns raw output
                if logits is False returns log-probabilities (log_softmax)"""

        output = self.net(
            x,
            pos,
            batch,
            edge_index=edge_index,
        )

        # linear layer on each fov feature vector
        if logits:
            return output
        else:
            return output.log_softmax(dim=-1)


class LocOnlyNetExplainHalf(LocOnlyNetExplain):
    """Second half of the neural network that takes in only localisations used for the explain module

    Args:
        config (dict): Dictionary containing the configuration for the network
        device (torch.device): Whether to run on cpu or gpu
        static (bool): If true edge indices are pre-calculated

    """

    def __init__(self, config, device, state_dict):
        super().__init__(config, device)
        self.load_state_dict(state_dict)
        self.final_transformer_half = self.net.transformers_down[-1]
        self.final_mlp_half = self.net.mlp_output

    def forward(self, x, edge_index, batch, pos, logits=True):
        """Method called when data runs through network

        Args:
            x (torch.tensor): Input feature that runs through the network
            edge_index (torch.tensor): Edge connections for the input graph
            batch (torch.tensor): Batch for the input graph
            pos (torch.tensor): Positions for the input graph
            logits (bool): Whether to output logits

        Returns:
            output (torch.tensor): If logits is True returns raw output
                if logits is False returns log-probabilities (log_softmax)"""

        x = self.final_transformer_half(
            x,
            pos,
            edge_index=edge_index,
        )

        x = global_max_pool(x, batch)

        output = self.final_mlp_half(x)

        # linear layer on each fov feature vector
        if logits:
            return output
        else:
            return output.log_softmax(dim=-1)
