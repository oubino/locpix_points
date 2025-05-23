# Load in final version of a model for each fold evaluate on data and measure number of correct for WT

# Imports
import argparse
import os
import numpy as np
import polars as pl

import torch
import torch_geometric.loader as L
import yaml
from torchsummary import summary

from locpix_points.data_loading import datastruc
from locpix_points.evaluation import evaluate
from locpix_points.models import model_choice
from locpix_points.scripts.process import load_pre_filter, minmax, minmaxpos
from sklearn.metrics import class_likelihood_ratios, accuracy_score

import torch
from torchmetrics import (
    Accuracy,
    #    F1Score,
    MetricCollection,
    #    Precision,
    Specificity,
    Recall,
    ROC,
    AUROC,
    #    AveragePrecision,
)

# from torchmetrics.classification import (
#    ConfusionMatrix,
#    MulticlassJaccardIndex,
#    StatScores,
# )

import warnings


def make_prediction_wt(
    file_map,
    model,
    loader,
    device,
    num_classes,
    explain=False,
    only_wt=False,
    repeats=25,
):
    """Make predictions using the model

    Args:
        file_map (df) : Contains map from the file to other variables
            e.g. mutation status
        model (pytorch geo model) : Model that will make predictiions
        loader (torch dataloader): Dataloader for the
            test dataset
        device (gpu or cpu): Device to evaluate the model
            on
        num_classes (int) : Number of classes in the dataset
        explain (bool) : Whether this is for an explain dataset
        only_wt (bool) : Whether to check performance only on the WT points
        repeats (int) : How many times to sample to generate prediction

    Returns:
        metrics (dict) : Dict containing the metrics
        metrics_roc (dict) : Dict containing ROC metrics
        file_list (list) : Names of files"""

    # currently set up metrics for binary
    # if want to change to multiclass look for lines need to change
    # MULTICLASS
    assert num_classes == 2

    model.to(device)

    metrics = MetricCollection(
        # ConfusionMatrix(task="binary"),     # MULTICLASS
        Recall(task="binary"),  # MULTICLASS
        Specificity(task="binary"),  # MULTICLASS
        # Precision(task="multiclass", num_classes=num_classes, average="none"),
        # F1Score(task="multiclass", num_classes=num_classes, average="none"),
        # MulticlassJaccardIndex(num_classes=num_classes, average="none"),
        Accuracy(task="binary"),  # MULTICLASS
        # StatScores(task="binary"),     # MULTICLASS
    ).to(device)

    metrics_roc = MetricCollection(
        ROC(task="binary"),  # MULTICLASS
        AUROC(task="binary"),  # MULTICLASS
        # AveragePrecision(task="multiclass", num_classes=num_classes, average="none"),
    ).to(device)

    # test data
    model.eval()
    stds = []
    file_list = []
    probs_list = []
    predictions_list = []
    gt_list = []
    for index, data in enumerate(loader):
        if only_wt:
            name = data.name[0]
            start_idx = name.find("_cell_")
            base_file = name[0:start_idx] + ".parquet"
            map = file_map.filter(pl.col("file_name") == base_file)
            muts = map[
                [
                    "kras1213_sr",
                    "kras61_sr",
                    "kras146_sr",
                    "nras1213_sr",
                    "nras61_sr",
                    "braf_sr",
                ]
            ].to_numpy()[0]
            if len([x for x in muts if not ((x == "W/T") or (x == "WT"))]) > 0:
                wt = "not-WT"
                continue

        with torch.no_grad():
            # note set to none is meant to have less memory footprint
            # move data to device
            data.to(device)

            file_list.append(data.name[0])  # only works cause batch size is 1

            # forward pass - with autocasting
            with torch.autocast(device_type="cuda"):
                if explain:
                    warnings.warn(
                        "Assuming that last layer is unnormalised\
                                  prob"
                    )
                    output = model(data.x, data.edge_index, data.batch)
                    output = output.log_softmax(dim=-1)
                    output = torch.exp(output)
                    print("output", output)
                    print("label", data.y)
                else:
                    output = []
                    for _ in range(repeats):
                        output.append(torch.exp(model(data)))
                    output = torch.stack(output)
                    std = torch.std(output, axis=0)
                    output = torch.mean(output, axis=0)
                    stds.append(std[0][0])
                    # output = torch.unsqueeze(output,0)

                # output is log softmax therefore convert back to prob
                metrics_roc.update(
                    output[:, 1], data.y
                )  # MULTICLASS - CHANGE output[:,1] -> output

                # prob list update
                probs_list.append(output[:, 1].item())

                # argmax predictions
                predictions = output.argmax(dim=1)
                predictions_list.append(predictions.item())
                metrics.update(predictions, data.y)

                gt_list.append(data.y.item())

    print("Average standard deviations: ", torch.mean(torch.stack(stds)))

    # metric over all batches
    metrics = metrics.compute()
    metrics_roc = metrics_roc.compute()

    if metrics["BinarySpecificity"].item() == 1:
        metrics["pos_lr"] = "infinity"
    else:
        metrics["pos_lr"] = metrics["BinaryRecall"].item() / (
            1 - metrics["BinarySpecificity"].item()
        )  # MULTICLASS

    return metrics, metrics_roc, file_list, probs_list, predictions_list, gt_list


def main(argv=None):
    # load config

    # parse arugments
    parser = argparse.ArgumentParser(description="Evaluating on WT")

    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="location of the project directory",
        required=True,
    )

    parser.add_argument(
        "-w",
        "--only_wt",
        action="store_true",
        help="if run on only wt",
    )

    parser.add_argument(
        "-n",
        "--model_name",
        action="store",
        type=str,
        required=False,
        help="name of the model in each fold otherwise assumes only one model present",
        default=None,
    )

    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        help="if provided then perform this number of repeats to generate the prediction",
        required=False,
        default=25,
    )

    parser.add_argument(
        "-f",
        "--final_test",
        action="store_true",
        help="final test",
    )

    parser.add_argument(
        "-m",
        "--map_file",
        action="store",
        type=str,
        required=True,
        help="location of the map file",
    )

    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        type=str,
        required=False,
        help="location of the config file",
    )

    parser.add_argument(
        "-fo",
        "--folds",
        type=int,
        required=False,
        help="number of folds",
        default=5,
    )

    parser.add_argument(
        "-RTS",
        "--evaluate_on_RTS",
        action="store_true",
        help="evaluate on reserved test set",
    )

    args = parser.parse_args(argv)
    file_map = pl.read_csv(args.map_file)

    config_loc = args.config_file

    if config_loc is None:
        config_loc = "config"

    project_directory = args.project_directory
    config_loc = os.path.join(project_directory, f"{config_loc}/evaluate.yaml")

    # load yaml
    with open(config_loc, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in config
    load_data_from_gpu = config["load_data_from_gpu"]
    eval_on_gpu = config["eval_on_gpu"]
    num_classes = config["num_classes"]

    assert num_classes == 2  # MULTICLASS

    # if data is on gpu then don't need to pin memory
    pin_memory = True

    # define device
    device = torch.device("cuda")

    only_wt = args.only_wt

    if args.final_test:
        raise NotImplementedError("Not implemented yet!")

    if not args.evaluate_on_RTS:
        final_test_file_list = []
        final_test_probs = []
        final_test_predictions = []
        final_test_gt = []

        train_aurocs = []
        val_aurocs = []
        test_aurocs = []

        train_balanced_accs = []
        val_balanced_accs = []
        test_balanced_accs = []

        train_plrs = []
        val_plrs = []
        test_plrs = []

        # For each fold
        for fold in range(args.folds):
            # Load in model
            processed_directory = os.path.join(
                project_directory, f"processed/fold_{fold}"
            )
            model_loc = os.path.join(project_directory, f"models/fold_{fold}")
            if args.model_name is None:
                assert len(os.listdir(model_loc)) == 1
                model_loc_sub = os.listdir(model_loc)[0]
            else:
                model_loc_sub = args.model_name
            model_loc = os.path.join(model_loc, model_loc_sub)

            train_folder = os.path.join(processed_directory, "train")
            val_folder = os.path.join(processed_directory, "val")
            test_folder = os.path.join(processed_directory, "test")

            train_files = pl.read_csv(os.path.join(train_folder, "file_map.csv"))[
                "file_name"
            ].to_list()
            val_files = pl.read_csv(os.path.join(val_folder, "file_map.csv"))[
                "file_name"
            ].to_list()
            test_files = pl.read_csv(os.path.join(test_folder, "file_map.csv"))[
                "file_name"
            ].to_list()
            assert len([x for x in test_files if x in train_files]) == 0
            assert len([x for x in test_files if x in val_files]) == 0
            assert len([x for x in val_files if x in train_files]) == 0

            if config["model"] == "loconlynet":
                # load in test dataset
                train_set = datastruc.LocDataset(
                    None,  # raw_loc_dir_root
                    train_folder,  # processed_dir_root
                    label_level=config["label_level"],  # label_level
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
                    range_xy=False,
                )

                # load in test dataset
                val_set = datastruc.LocDataset(
                    None,  # raw_loc_dir_root
                    val_folder,  # processed_dir_root
                    label_level=config["label_level"],  # label_level
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
                    range_xy=False,
                )

                # load in test dataset
                test_set = datastruc.LocDataset(
                    None,  # raw_loc_dir_root
                    test_folder,  # processed_dir_root
                    label_level=config["label_level"],  # label_level
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
                    range_xy=False,
                )

            elif config["model"] in [
                "locclusternet",
                "clusternet",
                "clustermlp",
                "locnetonly_pointnet",
                "locnetonly_pointtransformer",
            ]:
                train_set = datastruc.ClusterLocDataset(
                    None,  # raw_loc_dir_root
                    None,  # raw_cluster_dir_root
                    train_folder,  # processed_dir_root
                    label_level=config["label_level"],  # label_level
                    pre_filter=None,  # pre_filter
                    save_on_gpu=False,
                    transform=None,  # transform
                    pre_transform=None,  # pre_transform
                    loc_feat=None,
                    cluster_feat=None,
                    min_feat_locs=None,
                    max_feat_locs=None,
                    min_feat_clusters=None,
                    max_feat_clusters=None,
                    kneighboursclusters=None,
                    fov_x=None,
                    fov_y=None,
                    kneighbourslocs=None,
                    range_xy=False,
                )

                val_set = datastruc.ClusterLocDataset(
                    None,  # raw_loc_dir_root
                    None,  # raw_cluster_dir_root
                    val_folder,  # processed_dir_root
                    label_level=config["label_level"],  # label_level
                    pre_filter=None,  # pre_filter
                    save_on_gpu=False,
                    transform=None,  # transform
                    pre_transform=None,  # pre_transform
                    loc_feat=None,
                    cluster_feat=None,
                    min_feat_locs=None,
                    max_feat_locs=None,
                    min_feat_clusters=None,
                    max_feat_clusters=None,
                    kneighboursclusters=None,
                    fov_x=None,
                    fov_y=None,
                    kneighbourslocs=None,
                    range_xy=False,
                )

                test_set = datastruc.ClusterLocDataset(
                    None,  # raw_loc_dir_root
                    None,  # raw_cluster_dir_root
                    test_folder,  # processed_dir_root
                    label_level=config["label_level"],  # label_level
                    pre_filter=None,  # pre_filter
                    save_on_gpu=False,
                    transform=None,  # transform
                    pre_transform=None,  # pre_transform
                    loc_feat=None,
                    cluster_feat=None,
                    min_feat_locs=None,
                    max_feat_locs=None,
                    min_feat_clusters=None,
                    max_feat_clusters=None,
                    kneighboursclusters=None,
                    fov_x=None,
                    fov_y=None,
                    kneighbourslocs=None,
                    range_xy=False,
                )

            else:
                error_msg = config["model"]
                raise ValueError(f"Have not written for {error_msg}")

            train_loader = L.DataLoader(
                train_set,
                batch_size=1,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=0,
            )

            val_loader = L.DataLoader(
                val_set,
                batch_size=1,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=0,
            )

            test_loader = L.DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=0,
            )

            for _, data in enumerate(test_loader):
                first_item = data

            if config["model"] in [
                "locclusternet",
                "clusternet",
                "clustermlp",
                "locnetonly_pointnet",
                "locnetonly_pointtransformer",
            ]:
                dim = first_item["locs"].pos.shape[-1]
            elif config["model"] in ["loconlynet"]:
                dim = first_item.pos.shape[-1]
            else:
                raise ValueError("Model not defined")

            # initialise model
            model = model_choice(
                config["model"],
                # this should parameterise the chosen model
                config[config["model"]],
                dim=dim,
                device=device,
            )

            print("\n")
            print("Loading in best model")
            print("\n")
            model.load_state_dict(torch.load(model_loc, weights_only=False))
            model.to(device)

            # model summary
            print("\n")
            print("---- Model summary ----")
            print("\n")
            number_nodes = 1000  # this is just for summary, has no bearing on training
            summary(
                model,
                input_size=(test_set.num_node_features, number_nodes),
                batch_size=1,
            )

            repeats = args.repeats

            print("\n")
            print("---- Predict on train set... ----")
            print("\n")
            metrics, roc_metrics, train_file_list, _, _, _ = make_prediction_wt(
                file_map,
                model,
                train_loader,
                device,
                num_classes,
                explain=False,
                only_wt=only_wt,
                repeats=repeats,
            )

            print("AUROC: ", roc_metrics["BinaryAUROC"])  # MULTICLASS
            print("Accuracy: ", metrics["BinaryAccuracy"])  # MULTICLASS
            print("+ LR: ", metrics["pos_lr"])  # MULTICLASS
            print(
                "Balanced accuracy: ",
                0.5 * (metrics["BinaryRecall"] + metrics["BinarySpecificity"]),
            )

            train_aurocs.append(roc_metrics["BinaryAUROC"].cpu())
            train_balanced_accs.append(
                0.5
                * (metrics["BinaryRecall"].cpu() + metrics["BinarySpecificity"].cpu())
            )
            train_plrs.append(metrics["pos_lr"])

            print("\n")
            print("---- Predict on val set... ----")
            print("\n")
            metrics, roc_metrics, val_file_list, _, _, _ = make_prediction_wt(
                file_map,
                model,
                val_loader,
                device,
                num_classes,
                explain=False,
                only_wt=only_wt,
                repeats=repeats,
            )

            print("AUROC: ", roc_metrics["BinaryAUROC"])  # MULTICLASS
            print("Accuracy: ", metrics["BinaryAccuracy"])  # MULTICLASS
            print("+ LR: ", metrics["pos_lr"])  # MULTICLASS
            print(
                "Balanced accuracy: ",
                0.5 * (metrics["BinaryRecall"] + metrics["BinarySpecificity"]),
            )

            val_aurocs.append(roc_metrics["BinaryAUROC"].cpu())
            val_balanced_accs.append(
                0.5
                * (metrics["BinaryRecall"].cpu() + metrics["BinarySpecificity"].cpu())
            )
            val_plrs.append(metrics["pos_lr"])

            print("\n")
            print("---- Predict on test set... ----")
            print("\n")
            (
                metrics,
                roc_metrics,
                test_file_list,
                test_probs,
                test_predictions,
                test_gt,
            ) = make_prediction_wt(
                file_map,
                model,
                test_loader,
                device,
                num_classes,
                explain=False,
                only_wt=only_wt,
                repeats=repeats,
            )

            final_test_file_list.extend(test_file_list)
            final_test_probs.extend(test_probs)
            final_test_predictions.extend(test_predictions)
            final_test_gt.extend(test_gt)

            out = set(train_file_list) & set(val_file_list)
            assert not out
            out = set(train_file_list) & set(test_file_list)
            assert not out
            out = set(val_file_list) & set(test_file_list)
            assert not out

            print("AUROC: ", roc_metrics["BinaryAUROC"])  # MULTICLASS
            print("Accuracy: ", metrics["BinaryAccuracy"])  # MULTICLASS
            print("+ LR: ", metrics["pos_lr"])  # MULTICLASS
            print(
                "Balanced accuracy: ",
                0.5 * (metrics["BinaryRecall"] + metrics["BinarySpecificity"]),
            )

            test_aurocs.append(roc_metrics["BinaryAUROC"].cpu())
            test_balanced_accs.append(
                0.5
                * (metrics["BinaryRecall"].cpu() + metrics["BinarySpecificity"].cpu())
            )
            test_plrs.append(metrics["pos_lr"])

        print(final_test_file_list)
        print(final_test_probs)
        print(final_test_predictions)
        print(final_test_gt)

        print(
            "train AUROC: ",
            np.mean(np.array(train_aurocs)),
            " +- ",
            np.std(np.array(train_aurocs), ddof=1),
        )
        # print("train plr: ", np.mean(np.array(train_plrs)), " +- ", np.std(np.array(train_plrs), ddof=1))
        print(
            "train balanced acc: ",
            np.mean(np.array(train_balanced_accs)),
            " +- ",
            np.std(np.array(train_balanced_accs), ddof=1),
        )

        print(
            "val AUROC: ",
            np.mean(np.array(val_aurocs)),
            " +- ",
            np.std(np.array(val_aurocs), ddof=1),
        )
        # print("val plr: ", np.mean(np.array(val_plrs)), " +- ", np.std(np.array(val_plrs), ddof=1))
        print(
            "val balanced acc: ",
            np.mean(np.array(val_balanced_accs)),
            " +- ",
            np.std(np.array(val_balanced_accs), ddof=1),
        )

        print(
            "test AUROC: ",
            np.mean(np.array(test_aurocs)),
            " +- ",
            np.std(np.array(test_aurocs), ddof=1),
        )
        # print("test plr: ", np.mean(np.array(test_plrs)), " +- ", np.std(np.array(test_plrs), ddof=1))
        print(
            "test balanced acc: ",
            np.mean(np.array(test_balanced_accs)),
            " +- ",
            np.std(np.array(test_balanced_accs), ddof=1),
        )

        return (
            final_test_file_list,
            final_test_probs,
            final_test_predictions,
            final_test_gt,
        )

    else:
        print("Evaluate on RTS")
        warnings.warn("This set up is specific to my models etc.")

        RTS_probs = []
        RTS_predictions = []
        RTS_gt = []
        RTS_files = []

        RTS_aurocs = []
        RTS_balanced_accs = []
        RTS_plrs = []

        input_folder_train = os.path.join(project_directory, "preprocessed")
        file_directory = os.path.join(input_folder_train, "featextract/locs")

        input_folder_RTS = os.path.join(project_directory, "RTS/preprocessed")

        project_directory = args.project_directory
        with open(
            os.path.join(project_directory, f"{args.config_file}/process.yaml"), "r"
        ) as ymlfile:
            process_config = yaml.safe_load(ymlfile)

        # For each fold
        for fold in range(args.folds):
            train_list_path = os.path.join(
                project_directory, f"processed/fold_{fold}/train/pre_filter.pt"
            )
            train_list = load_pre_filter(train_list_path)

            min_feat_locs, max_feat_locs = minmax(
                process_config, "loc_feat", file_directory, train_list
            )

            if process_config["normalise"] == "per_dataset":
                # calculate xy range
                range_xy = minmaxpos(file_directory, train_list)
            elif process_config["normalise"] == "per_item":
                range_xy = None
            else:
                raise NotImplementedError("Normalise should be per-item or per-dataset")

            file_directory = os.path.join(input_folder_train, "featextract/clusters")
            min_feat_clusters, max_feat_clusters = minmax(
                process_config, "cluster_feat", file_directory, train_list
            )

            if "superclusters" in process_config.keys():
                superclusters = True
            else:
                superclusters = False

            RTS_folder = os.path.join(project_directory, "RTS/test")
            # if output directory not present create it
            if not os.path.exists(RTS_folder):
                os.makedirs(RTS_folder)

            # Load in model
            model_loc = os.path.join(project_directory, f"models/fold_{fold}")
            model_loc = os.path.join(model_loc, args.model_name)

            RTS_set = datastruc.ClusterLocDataset(
                os.path.join(input_folder_RTS, "featextract/locs"),
                os.path.join(input_folder_RTS, "featextract/clusters"),
                RTS_folder,
                process_config["label_level"],
                None,  # prefilter
                process_config["save_on_gpu"],
                None,  # transform
                None,  # pre-transform
                process_config["loc_feat"],
                process_config["cluster_feat"],
                min_feat_locs,
                max_feat_locs,
                min_feat_clusters,
                max_feat_clusters,
                process_config["kneighboursclusters"],
                process_config["fov_x"],
                process_config["fov_y"],
                kneighbourslocs=process_config["kneighbourslocs"],
                superclusters=superclusters,
                range_xy=range_xy,
            )

            RTS_loader = L.DataLoader(
                RTS_set,
                batch_size=1,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=0,
            )

            for _, data in enumerate(RTS_loader):
                first_item = data

            dim = first_item["locs"].pos.shape[-1]

            # initialise model
            model = model_choice(
                config["model"],
                # this should parameterise the chosen model
                config[config["model"]],
                dim=dim,
                device=device,
            )

            print("\n")
            print("Loading in best model")
            print("\n")
            model.load_state_dict(torch.load(model_loc, weights_only=False))
            model.to(device)

            # model summary
            print("\n")
            print("---- Model summary ----")
            print("\n")
            number_nodes = 1000  # this is just for summary, has no bearing on training
            summary(
                model,
                input_size=(RTS_set.num_node_features, number_nodes),
                batch_size=1,
            )

            repeats = args.repeats

            print("\n")
            print("---- Predict on RTS set... ----")
            print("\n")
            (
                RTS_metrics,
                RTS_roc_metrics,
                RTS_files_,
                RTS_probs_,
                RTS_predictions_,
                RTS_gt_,
            ) = make_prediction_wt(
                file_map,
                model,
                RTS_loader,
                device,
                num_classes,
                explain=False,
                only_wt=only_wt,
                repeats=repeats,
            )

            RTS_files.append(RTS_files_)
            RTS_probs.append(RTS_probs_)
            RTS_predictions.append(RTS_predictions_)
            RTS_gt.append(RTS_gt_)

            print("AUROC: ", RTS_roc_metrics["BinaryAUROC"])  # MULTICLASS
            print("Accuracy: ", RTS_metrics["BinaryAccuracy"])  # MULTICLASS
            print("+ LR: ", RTS_metrics["pos_lr"])  # MULTICLASS
            print(
                "Balanced accuracy: ",
                0.5 * (RTS_metrics["BinaryRecall"] + RTS_metrics["BinarySpecificity"]),
            )

            RTS_aurocs.append(RTS_roc_metrics["BinaryAUROC"].cpu())
            RTS_balanced_accs.append(
                0.5
                * (
                    RTS_metrics["BinaryRecall"].cpu()
                    + RTS_metrics["BinarySpecificity"].cpu()
                )
            )
            RTS_plrs.append(RTS_metrics["pos_lr"])

        # print(RTS_probs)
        # print(RTS_predictions)
        # print(RTS_gt)

        print(
            "RTS AUROC: ",
            np.mean(np.array(RTS_aurocs)),
            " +- ",
            np.std(np.array(RTS_aurocs), ddof=1),
        )
        # print("RTS plr: ", np.mean(np.array(RTS_plrs)), " +- ", np.std(np.array(RTS_plrs), ddof=1))
        print(
            "RTS balanced acc: ",
            np.mean(np.array(RTS_balanced_accs)),
            " +- ",
            np.std(np.array(RTS_balanced_accs), ddof=1),
        )

        return RTS_files, RTS_probs, RTS_predictions, RTS_gt


if __name__ == "__main__":
    main()
