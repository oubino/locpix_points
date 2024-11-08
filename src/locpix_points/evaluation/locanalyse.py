"""Feature analysis module

Module takes in the processed localisations.

Then rather than passing through the model as usual - calculating the edge index dynamically

We construct the edge index statically for the initial localisations.

We mandate that all ratios set to 1.

We can then perform subgraphX and attention (if relevant) analysis.

We could also implement other point cloud based explainability methods.

Config file at top specifies the analyses we want to run"""

import argparse
import json
import os
import time
from typing import List
import yaml

from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.evaluation import XCollector
from torcheval.metrics import MulticlassConfusionMatrix, MulticlassAccuracy
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from locpix_points.data_loading import datastruc
from locpix_points.models import loc_only_nets
from locpix_points.evaluation.featanalyse import (
    visualise_explanation,
    custom_fidelity_measure,
)

import polars as pl
import torch
import torch_geometric.loader as L
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch import Tensor


def analyse_locs(project_directory, config, final_test, automatic):
    """Analyse the localisations

    Args:
        project_directory (str): Location of the project directory
        config (dict): Configuration for this script
        final_test (bool): If true is on final test
        automatic (bool): If true automatically determine model

    Returns:
        train_set (dataset): Training dataset
        train_map (dict): Map from file names to ids for train set
        test_set (dataset): Test dataset
        test_map (dict): Map from file names to ids for test set
        model (pyg model): PyG model that we are explaining
        model_type (str): Type of PyG model
        config (dict): Configuration for the explain
        device (torch device): Device to run explanation on

    Raises:
        ValueError: If device specified is neither cpu or gpu OR
            if attention to examine is not correctly specified OR
            if encoder not loc or cluster or fov
    """

    # ----------------------------

    if config["device"] == "gpu":
        device = torch.device("cuda")
    elif config["device"] == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("Device should be cpu or gpu")

    # load in gt_label_map
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        gt_label_map = metadata["gt_label_map"]

    gt_label_map = {int(key): val for key, val in gt_label_map.items()}

    model_type = config["model"]

    # only works for locclusternet at the moment
    assert model_type == "loconlynet"

    # initialise model
    model = loc_only_nets.LocOnlyNetExplain(
        config[model_type],
        device=device,
    )

    # load in best model
    print("\n")
    print("Loading in best model")
    print("\n")
    if not final_test:
        # needs to be from same fold as below
        fold = config["fold"]
    model_name = config["model_name"]
    if not automatic:
        if not final_test:
            model_loc = os.path.join(
                project_directory, "models", f"fold_{fold}", model_name
            )
        else:
            model_loc = os.path.join(project_directory, "models", model_name)
    elif automatic:
        if not final_test:
            model_dir = os.path.join(project_directory, "models", f"fold_{fold}")
        else:
            model_dir = os.path.join(project_directory, "models")
        model_list = os.listdir(model_dir)
        assert len(model_list) == 1
        model_name = model_list[0]
        model_loc = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(model_loc))
    model.to(device)
    model.eval()

    if not final_test:
        train_folder = os.path.join(
            project_directory, "processed", f"fold_{fold}", "train"
        )
        val_folder = os.path.join(project_directory, "processed", f"fold_{fold}", "val")
        test_folder = os.path.join(
            project_directory, "processed", f"fold_{fold}", "test"
        )
    else:
        train_folder = os.path.join(project_directory, "processed", "train")
        val_folder = os.path.join(project_directory, "processed", "val")
        test_folder = os.path.join(project_directory, "processed", "test")

    # initialise train/validation and test sets
    train_set = datastruc.LocDataset(
        None,  # raw_loc_dir_root
        train_folder,  # processed_dir_root
        label_level=None,
        pre_filter=None,  # pre_filter
        save_on_gpu=False,  # gpu
        transform=None,  # transform
        pre_transform=None,  # pre_transform
        feat=None,
        min_feat=None,
        max_feat=None,
        fov_x=None,
        fov_y=None,
        kneighbours=None,
    )

    # load in val dataset
    val_set = datastruc.LocDataset(
        None,  # raw_loc_dir_root
        val_folder,  # processed_dir_root
        label_level=None,
        pre_filter=None,  # pre_filter
        save_on_gpu=False,  # gpu
        transform=None,
        pre_transform=None,
        feat=None,
        min_feat=None,
        max_feat=None,
        fov_x=None,
        fov_y=None,
        kneighbours=None,
    )

    # load in test dataset
    test_set = datastruc.LocDataset(
        None,  # raw_loc_dir_root
        test_folder,  # processed_dir_root
        label_level=None,
        pre_filter=None,  # pre_filter
        save_on_gpu=False,  # gpu
        transform=None,  # transform
        pre_transform=None,  # pre_transform
        feat=None,
        min_feat=None,
        max_feat=None,
        fov_x=None,
        fov_y=None,
        kneighbours=None,
    )

    train_map = pl.read_csv(
        os.path.join(train_set.processed_dir, "file_map.csv")
    ).rows_by_key("file_name", unique=True)
    val_map = pl.read_csv(
        os.path.join(val_set.processed_dir, "file_map.csv")
    ).rows_by_key("file_name", unique=True)
    test_map = pl.read_csv(
        os.path.join(test_set.processed_dir, "file_map.csv")
    ).rows_by_key("file_name", unique=True)

    # aggregate cluster features into collated df
    if not final_test:
        train_set = torch.utils.data.ConcatDataset([train_set, val_set, test_set])
        test_set = None
        # combine train/val/test maps
        train_len = train_map[list(train_map)[-1]] + 1
        val_map = {key: val + train_len for key, val in val_map.items()}
        val_len = val_map[list(val_map)[-1]] + 1
        test_map = {key: val + val_len for key, val in test_map.items()}
        train_map = train_map | val_map
        train_map = train_map | test_map
        test_map = None
    else:
        train_set = torch.utils.data.ConcatDataset([train_set, val_set])
        # combine train and val map
        train_len = train_map[list(train_map)[-1]] + 1
        val_map = {key: val + train_len for key, val in val_map.items()}
        train_map = train_map | val_map
        test_set = test_set

    return train_set, train_map, test_set, test_map, model, model_type, config, device


def explain(
    file_names,
    map,
    dataset,
    model,
    model_type,
    config,
    device,
    type=None,
    subgraph_config=None,
    intermediate=False,
):
    """Wrapper for algorithms to generate the explanation

    Args:
        file_names (list): List of files that need explaining
        map (dictionary): Map from the file names to their IDs
        dataset (pyg dataset): The dataset that contains the data to be explained
        model (pyg model): Model that needs explaining
        model_type (str): The type of the model that needs explaining
        config (dict): Configuration for the explanation
        device (torch device): Device to run the explanation on
        type (str): Type of explanation algorithm
        subgraph_config (dict): Configuration for SubgraphX
        intermediate (bool): For subgraphX whether to look at intermediate graph

    Returns:
        output (list): List of outputs for each file

    Raises:
        ValueError: Temporary measure as can't handle PointNet
    """

    outputs = []

    for file_name in file_names:
        idx = map[file_name]
        data = dataset.__getitem__(idx)
        assert data.name == file_name
        with torch.no_grad():
            data.to(device)

            if data.x is None:
                data.x = torch.ones((data.pos.shape[0], 1), device=device)

            with torch.autocast(device_type="cuda"):
                if config[model_type]["conv_type"] == "pointnet":
                    error = (
                        "This is hard as from the very first edge index it randomly samples and since it uses bipartite pointnetconv"
                        "there isn't an obvious way to do this"
                    )
                    raise ValueError(error)

                elif config[model_type]["conv_type"] == "pointtransformer":
                    data.edge_index = knn_graph(
                        data.pos, k=config[model_type]["k"], batch=data.batch
                    )

                    logits = model(
                        data.x, data.edge_index, data.batch, data.pos, logits=True
                    )

                    prediction = logits.argmax(-1).item()

                    print("GT: ", data.y.item())
                    print("Prediction: ", prediction)

            if type == "subgraphx":
                outputs.append(
                    subgraph_eval(
                        data,
                        prediction,
                        model,
                        config[model_type],
                        subgraph_config,
                        device,
                        intermediate=intermediate,
                    )
                )

            elif type == "attention":
                outputs.append(attention_eval(data, model))

    return outputs


def subgraph_eval(
    data, prediction, model, model_config, subgraph_config, device, intermediate=False
):
    """SubgraphX algorithm

    Args:
        data (pyg dataitem): Dataitem to explain
        prediction (torch tensor): Prediction for the dataitem
        model (PyG model): Model to explain
        model_config (dict): Configuration for the model to be explained
        subgraph_config (dict): Configuration for the subgraphx algorithm
        device (torch device): Device to run explanation on
        intermediate (bool): Whether to evaluate intermediate graph

    Returns:
        subgraph (PyG graph): The induced subgraph from the important structure
        complement (PyG graph): The complement to the subgraph
        data (pyg dataitem): Item being explained
        node_imp (tensor): Importance of each tensor"""

    if intermediate:
        # store intermediate values
        x_intermediate = []
        edge_index_intermediate = []
        batch_intermediate = []
        pos_intermediate = []

        def hook_output(module, input, output):
            batch_intermediate.append(output[2])

        def hook_input(module, input, output):
            x_intermediate.append(input[0])
            pos_intermediate.append(input[1])
            edge_index_intermediate.append(input[2])

        hook_handles = []

        hook_handles.append(
            model.net.transition_down[-1].register_forward_hook(hook_output)
        )
        hook_handles.append(
            model.net.transformers_down[-1].register_forward_hook(hook_input)
        )

        model(data.x, data.edge_index, data.batch, data.pos, logits=False)

        for handle in hook_handles:
            handle.remove()

        # get the intermediate graph we will run through half the model
        data = Data(
            x=x_intermediate[0],
            pos=pos_intermediate[0],
            edge_index=edge_index_intermediate[0],
            batch=batch_intermediate[0],
        )
        data.to(device)

        model = loc_only_nets.LocOnlyNetExplainHalf(
            model_config,
            device,
            model.state_dict(),
        )
        model.to(device)
        model.eval()

    explainer = SubgraphX(
        model,
        num_classes=subgraph_config["num_classes"],
        device=device,
        explain_graph=True,
        rollout=subgraph_config["rollout"],
        min_atoms=subgraph_config["min_atoms"],
        c_puct=subgraph_config["c_puct"],
        expand_atoms=subgraph_config["expand_atoms"],
        high2low=subgraph_config["high2low"],
        local_radius=subgraph_config["local_radius"],
        sample_num=subgraph_config["sample_num"],
        reward_method=subgraph_config["reward_method"],
        subgraph_building_method=subgraph_config["subgraph_building_method"],
        vis=False,
    )

    # generate explanation for the graph
    _, explanation_results, related_preds = explainer(
        data.x,
        data.edge_index,
        forward_kwargs={
            "batch": data.batch,
            "pos": data.pos,
            "logits": True,
        },
        max_nodes=subgraph_config["max_nodes"],
    )

    # process explanation results
    explanation_results = explanation_results[prediction]
    explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)
    tree_node_x = find_closest_node_result(
        explanation_results, max_nodes=subgraph_config["max_nodes"]
    )

    # generate metrics for explanation
    nodelist = tree_node_x.coalition
    node_imp = torch.zeros(len(data.pos))
    node_imp[nodelist] = 1.0
    x_collector = XCollector()
    x_collector.collect_data(tree_node_x.coalition, related_preds, label=prediction)

    # print metrics for explanation
    print(f"Sparsity: {x_collector.sparsity:.4f}")
    print(f"Accuracy: {x_collector.accuracy:.4f}")
    print(f"Stability: {x_collector.stability:.4f}")

    # evaluate explanation
    visualise_explanation(
        data.pos,
        data.edge_index,
        node_imp=node_imp.to(device),
        edge_imp=None,
    )

    # alternative fidelity measures
    subgraph, complement = custom_fidelity_measure(
        model, data, node_imp, "node", device, batch=data.batch
    )

    return subgraph, complement, data, node_imp


def attention_eval(data, model):
    """Examine attention weights

    Args:
        data (pyg dataitem): Dataitem to explain
        model (PyG model): Model to explain

    Returns:
        positions (PyG graph): Positions of the data item localisations
        edge_indices (PyG graph): Edge indices of the data item being explained
        alpha (pyg dataitem): Attention coefficients

    Raises:
        ValueError: If can't find any attention coefficients"""

    alphas: List[Tensor] = []
    edge_indices: List[Tensor] = []
    positions: List[Tensor] = []

    def hook(module, msg_kwargs, out):
        if getattr(module, "_alpha", None) is not None:
            alphas.append(module._alpha.detach())
        else:
            raise ValueError("alphas not present")

    def hook_inputs(module, inputs):
        edge_indices.append(inputs[0])
        positions.append(inputs[2]["pos"][0])

    hook_handles = []
    hook_handles_inputs = []

    for module in model.modules():  # Register message forward hooks:
        for name, module in module.named_children():
            if name == "transformer":
                hook_handles.append(module.register_message_forward_hook(hook))
                hook_handles_inputs.append(
                    module.register_propagate_forward_pre_hook(hook_inputs)
                )

    model(data.x, data.edge_index, data.batch, data.pos, logits=False)

    for handle in hook_handles:  # Remove hooks:
        handle.remove()

    if len(alphas) == 0:
        raise ValueError(
            "Could not collect any attention coefficients. "
            "Please ensure that your model is using "
            "attention-based GNN layers."
        )

    for i, alpha in enumerate(alphas):
        alpha = alpha[: edge_indices[i].size(1)]  # Respect potential self-loops.
        if alpha.dim() == 2:
            alpha = getattr(torch, "max")(alpha, dim=-1)
            if isinstance(alpha, tuple):  # Respect `torch.max`:
                alpha = alpha[0]
        elif alpha.dim() > 2:
            raise ValueError(
                f"Can not reduce attention coefficients of "
                f"shape {list(alpha.size())}"
            )
        alphas[i] = alpha

    return positions, edge_indices, alphas


### - ARCHIVE ---


# def main(argv=None):
#    """Main script for the module with variable arguments
#
#    Args:
#        argv : Custom arguments to run script with"""
#
#    # parse arugments
#    parser = argparse.ArgumentParser(description="Analyse features")
#
#    parser.add_argument(
#        "-i",
#        "--project_directory",
#        action="store",
#        type=str,
#        help="location of the project directory",
#        required=True,
#    )
#
#    parser.add_argument(
#        "-c",
#        "--config",
#        action="store",
#        type=str,
#        help="the location of the .yaml configuaration file\
#                             for evaluating",
#        required=True,
#    )
#
#    parser.add_argument(
#        "-a",
#        "--automatic",
#        action="store_true",
#        help="if present then there should be only one model present in the folder"
#        "which we load in",
#    )
#
#    parser.add_argument(
#        "-f",
#        "--final_test",
#        action="store_true",
#        help="if specified then running final test",
#    )
#
#    parser.add_argument(
#        "-if",
#        "--files_to_test_on",
#        action="store",
#        nargs="+",
#        help="stores list of names of files want to evaluate",
#        required=True,
#    )
#
#    args = parser.parse_args(argv)
#
#    project_directory = args.project_directory
#
#    # load config
#    with open(args.config, "r") as ymlfile:
#        config = yaml.safe_load(ymlfile)
#    label_map = config["label_map"]
#
#    metadata_path = os.path.join(project_directory, "metadata.json")
#    with open(
#        metadata_path,
#    ) as file:
#        metadata = json.load(file)
#        # add time ran this script to metadata
#        file = os.path.basename(__file__)
#        if file not in metadata:
#            metadata[file] = time.asctime(time.gmtime(time.time()))
#        else:
#            print("Overwriting metadata...")
#            metadata[file] = time.asctime(time.gmtime(time.time()))
#        with open(metadata_path, "w") as outfile:
#            json.dump(metadata, outfile)
#
#    # make output folder
#    output_folder = os.path.join(project_directory, "output")
#    if not os.path.exists(output_folder):
#        os.makedirs(output_folder)
#
#    # ---- Analyse loc features -------
#    analyse_locs(project_directory, config, args)


# if __name__ == "__main__":
#    main()
