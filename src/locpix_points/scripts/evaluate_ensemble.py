# Load in final version of a model for each fold evaluate on data and measure number of correct for WT

# Imports
import argparse
import os
import polars as pl

import torch
import torch_geometric.loader as L
import yaml
from torchsummary import summary

from locpix_points.data_loading import datastruc
from locpix_points.evaluation import evaluate
from locpix_points.models import model_choice
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

                # argmax predictions
                predictions = output.argmax(dim=1)
                predictions_list.append(predictions.item())
                metrics.update(predictions, data.y)

                gt_list.append(data.y.item())

    print("Average standard deviations: ", torch.mean(torch.stack(stds)))

    # metric over all batches
    metrics = metrics.compute()
    metrics_roc = metrics_roc.compute()

    metrics["pos_lr"] = metrics["BinaryRecall"].item() / (
        1 - metrics["BinarySpecificity"].item()
    )  # MULTICLASS

    return metrics, metrics_roc, file_list, predictions_list, gt_list


def main():
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

    args = parser.parse_args()
    file_map = pl.read_csv(args.map_file)

    project_directory = args.project_directory
    config_loc = os.path.join(project_directory, "config/evaluate.yaml")

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

    final_test_file_list = []
    final_test_predictions = []
    final_test_gt = []

    # For each fold
    for fold in range(5):
        # Load in model
        processed_directory = os.path.join(project_directory, f"processed/fold_{fold}")
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
        model.load_state_dict(torch.load(model_loc))
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
        metrics, roc_metrics, train_file_list, _, _ = make_prediction_wt(
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

        print("\n")
        print("---- Predict on val set... ----")
        print("\n")
        metrics, roc_metrics, val_file_list, _, _ = make_prediction_wt(
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

        print("\n")
        print("---- Predict on test set... ----")
        print("\n")
        (
            metrics,
            roc_metrics,
            test_file_list,
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

    print(final_test_file_list)
    print(final_test_predictions)


if __name__ == "__main__":
    main()
