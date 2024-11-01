"""PointTransformer


PointTransformer adapted from
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py
and
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_segmentation.py

Originally in
PointTransformer http://arxiv.org/abs/2012.09164

"""

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    global_max_pool,
    knn,
    knn_graph,
    knn_interpolate,
)
from torch_geometric.utils import scatter, contains_self_loops


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, pos_nn_layers, attn_nn_layers):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        # !TODO: Below needs to depend on the dimensions and in paper it says
        # one ReLU non-linearity but the MLP has one after each linear layer...
        # HARDCODE
        self.pos_nn = MLP(
            [dim, pos_nn_layers, out_channels], plain_last=False, act="relu"
        )  # BN

        # HARDCODE
        self.attn_nn = MLP(  # BN
            [out_channels, attn_nn_layers, out_channels],
            plain_last=False,
            act="relu",
        )

        self.transformer = PointTransformerConv(
            in_channels,
            out_channels,
            pos_nn=self.pos_nn,
            attn_nn=self.attn_nn,
            aggr="max",
            add_self_loops=True,
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        assert not contains_self_loops(edge_index)
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionUp(torch.nn.Module):
    """
    Reduce features dimensionnality and interpolate back to higher
    resolution and cardinality
    """

    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.mlp_sub = MLP(
            [in_channels, out_channels],
            plain_last=False,
            act="relu",
        )
        self.mlp = MLP(
            [out_channels, out_channels],
            plain_last=False,
            act="relu",
        )
        self.k = k

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(
            x_sub, pos_sub, pos, k=self.k, batch_x=batch_sub, batch_y=batch
        )

        x = self.mlp(x) + x_interpolated

        return x


class TransitionDown(torch.nn.Module):
    """
    Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality
    """

    def __init__(self, in_channels, out_channels, ratio, k):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP(
            [in_channels, out_channels],
            plain_last=False,
            norm="instance_norm",
            act="relu",
        )  # BN

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster (they say cluster as each point comes to represent
        # a cluster despite being just points in the first layer) t
        sub_batch = batch[id_clusters] if batch is not None else None
        # add one to nearest neighs as nearest neighs includes itself
        # note this includes self loops
        id_k_neighbor = knn(
            pos, pos[id_clusters], k=self.k + 1, batch_x=batch, batch_y=sub_batch
        )

        # transformation of features through a simple MLP
        x = self.mlp(x, batch=batch)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(
            x[id_k_neighbor[1]],
            id_k_neighbor[0],
            dim=0,
            dim_size=id_clusters.size(0),
            reduce="max",
        )

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformerEmbedding(torch.nn.Module):
    def __init__(self, config, dim=2):
        super().__init__()

        self.name = "PointTransformerClassifier"
        self.k = config["k"]
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        dim_model = config["dim_model"]
        output_mlp_layers = config["output_mlp_layers"]
        ratio = config["ratio"]
        pos_nn_layers = config["pos_nn_layers"]
        attn_nn_layers = config["attn_nn_layers"]
        dropout = config["dropout"]

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP(
            [in_channels, dim_model[0]],
            plain_last=False,
            act="relu",
        )  # BN

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
            dim=dim,
            pos_nn_layers=pos_nn_layers,
            attn_nn_layers=attn_nn_layers,
        )
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i],
                    out_channels=dim_model[i + 1],
                    ratio=ratio[i],
                    k=self.k,
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1],
                    out_channels=dim_model[i + 1],
                    dim=dim,
                    pos_nn_layers=pos_nn_layers,
                    attn_nn_layers=attn_nn_layers,
                )
            )

        # class score computation
        # this has plain last = True by default
        self.mlp_output = MLP(  # BN
            [dim_model[-1], output_mlp_layers, out_channels],
            norm=None,
            act="relu",
            dropout=dropout,
        )

    def forward(self, x, pos, batch=None, edge_index=None):
        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        if edge_index is None:
            edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return out


class Segmenter(torch.nn.Module):
    def __init__(self, config, dim=2):
        super().__init__()

        self.name = "PointTransformerSegmenter"
        self.k = config["k"]
        k_up = config["k_up"]
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        dim_model = config["dim_model"]
        ratio = config["ratio"]
        output_mlp_layers = config["output_mlp_layers"]
        pos_nn_layers = config["pos_nn_layers"]
        attn_nn_layers = config["attn_nn_layers"]

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
            dim=dim,
            pos_nn_layers=pos_nn_layers,
            attn_nn_layers=attn_nn_layers,
        )

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):
            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i],
                    out_channels=dim_model[i + 1],
                    ratio=ratio,
                    k=self.k,
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1],
                    out_channels=dim_model[i + 1],
                    dim=dim,
                    pos_nn_layers=pos_nn_layers,
                    attn_nn_layers=attn_nn_layers,
                )
            )

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(
                    in_channels=dim_model[i + 1], out_channels=dim_model[i], k=k_up
                )
            )

            self.transformers_up.append(
                TransformerBlock(
                    in_channels=dim_model[i],
                    out_channels=dim_model[i],
                    dim=dim,
                    pos_nn_layers=pos_nn_layers,
                    attn_nn_layers=attn_nn_layers,
                )
            )

        # summit layers
        self.mlp_summit = MLP(
            [dim_model[-1], dim_model[-1]], norm=None, plain_last=False
        )

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
            dim=dim,
            pos_nn_layers=pos_nn_layers,
            attn_nn_layers=attn_nn_layers,
        )

        # class score computation
        self.mlp_output = MLP(
            [dim_model[0], output_mlp_layers, out_channels], norm=None
        )

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](
                x=out_x[-i - 2],
                x_sub=x,
                pos=out_pos[-i - 2],
                pos_sub=out_pos[-i - 1],
                batch_sub=out_batch[-i - 1],
                batch=out_batch[-i - 2],
            )

            edge_index = knn_graph(out_pos[-i - 2], k=self.k, batch=out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)
