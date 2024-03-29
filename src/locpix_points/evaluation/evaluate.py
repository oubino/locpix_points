"""Evaluation module

This contains functions for evaluating the models
"""

import torch
from torchmetrics import (
    Accuracy,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
    ROC,
    AUROC,
)
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
)
import warnings


def make_prediction(model, optimiser, train_loader, val_loader, device, num_classes):
    """Make predictions using the model

    Args:
        model (pytorch geo model) : Model that will make predictiions
        optimiser (pytorch optimiser) : Optimiser used in training
        train_loader (torch dataloader): Dataloader for the
            training dataset
        val_loader (torch dataloader): Dataloader for the
            validation dataset
        device (gpu or cpu): Device to train the model
            on
        num_classes (int) : Number of classes in the dataset

    Returns:
        metrics (dict) : Dict containing the metrics"""

    model.to(device)

    train_metrics = MetricCollection(
        MulticlassConfusionMatrix(num_classes=num_classes),
        Recall(task="multiclass", num_classes=num_classes, average="none"),
        Precision(task="multiclass", num_classes=num_classes, average="none"),
        F1Score(task="multiclass", num_classes=num_classes, average="none"),
        MulticlassJaccardIndex(num_classes=num_classes, average="none"),
        Accuracy(task="multiclass", num_classes=num_classes, average="none"),
    ).to(device)

    train_metrics_roc = MetricCollection(
        ROC(task="multiclass", num_classes=num_classes),
        AUROC(task="multiclass", num_classes=num_classes, average="none"),
    ).to(device)

    val_metrics = MetricCollection(
        MulticlassConfusionMatrix(num_classes=num_classes),
        Recall(task="multiclass", num_classes=num_classes, average="none"),
        Precision(task="multiclass", num_classes=num_classes, average="none"),
        F1Score(task="multiclass", num_classes=num_classes, average="none"),
        MulticlassJaccardIndex(num_classes=num_classes, average="none"),
        Accuracy(task="multiclass", num_classes=num_classes, average="none"),
    ).to(device)

    val_metrics_roc = MetricCollection(
        ROC(task="multiclass", num_classes=num_classes),
        AUROC(task="multiclass", num_classes=num_classes, average="none"),
    ).to(device)

    warnings.warn(
        "Assumes that the output from the network is log probability"
        "for the ROC curves"
    )

    # training data
    model.eval()
    for index, data in enumerate(train_loader):
        with torch.no_grad():
            # note set to none is meant to have less memory footprint
            optimiser.zero_grad(set_to_none=True)

            # move data to device
            data.to(device)

            # forward pass - with autocasting
            with torch.autocast(device_type="cuda"):
                output = model(data)

                # output is log softmax therefore convert back to prob
                train_metrics_roc.update(torch.exp(output), data.y)

                # argmax predictions
                train_predictions = output.argmax(dim=1)
                train_metrics.update(train_predictions, data.y)

    for index, data in enumerate(val_loader):
        with torch.no_grad():
            # note set to none is meant to have less memory footprint
            optimiser.zero_grad(set_to_none=True)

            # move data to device
            data.to(device)

            # forward pass - with autocasting
            with torch.autocast(device_type="cuda"):
                output = model(data)

                # output is log softmax therefore convert back to prob
                val_metrics_roc.update(torch.exp(output), data.y)

                # argmax predictions
                val_predictions = output.argmax(dim=1)
                val_metrics.update(val_predictions, data.y)

    # metric over all batches
    train_metrics = train_metrics.compute()
    val_metrics = val_metrics.compute()
    train_metrics_roc = train_metrics_roc.compute()
    val_metrics_roc = val_metrics_roc.compute()

    # output metrics
    metrics = {}
    roc_metrics = {}

    # make into format acceptable to wandb
    for i in range(num_classes):
        metrics[f"TrainRecall_{i}"] = train_metrics["MulticlassRecall"][i]
        metrics[f"ValRecall_{i}"] = val_metrics["MulticlassRecall"][i]
        metrics[f"TrainPrecision_{i}"] = train_metrics["MulticlassPrecision"][i]
        metrics[f"ValPrecision_{i}"] = val_metrics["MulticlassPrecision"][i]
        metrics[f"TrainF1Score_{i}"] = train_metrics["MulticlassF1Score"][i]
        metrics[f"ValF1Score_{i}"] = val_metrics["MulticlassF1Score"][i]
        metrics[f"TrainJaccardIndex_{i}"] = train_metrics["MulticlassJaccardIndex"][i]
        metrics[f"ValJaccardIndex_{i}"] = val_metrics["MulticlassJaccardIndex"][i]
        metrics[f"TrainAccuracy_{i}"] = train_metrics["MulticlassAccuracy"][i]
        metrics[f"ValAccuracy_{i}"] = val_metrics["MulticlassAccuracy"][i]

        roc_metrics[f"TrainFPR_{i}"] = train_metrics_roc["MulticlassROC"][0][i]
        roc_metrics[f"ValFPR_{i}"] = val_metrics_roc["MulticlassROC"][0][i]
        roc_metrics[f"TrainTPR_{i}"] = train_metrics_roc["MulticlassROC"][1][i]
        roc_metrics[f"ValTPR_{i}"] = val_metrics_roc["MulticlassROC"][1][i]
        roc_metrics[f"TrainThreshold_{i}"] = train_metrics_roc["MulticlassROC"][2][i]
        roc_metrics[f"ValThreshold_{i}"] = val_metrics_roc["MulticlassROC"][2][i]
        metrics[f"TrainAUROC_{i}"] = train_metrics_roc["MulticlassAUROC"][i]
        metrics[f"ValAUROC_{i}"] = val_metrics_roc["MulticlassAUROC"][i]

        for j in range(num_classes):
            metrics[f"train_actual_{i}_pred_{j}"] = train_metrics[
                "MulticlassConfusionMatrix"
            ][i][j]
            metrics[f"val_actual_{i}_pred_{j}"] = val_metrics[
                "MulticlassConfusionMatrix"
            ][i][j]

    if num_classes == 2:
        # ISSUE: This metric is specific only to when have two classes
        for split in ["train", "val"]:
            tp = metrics[f"{split}_actual_1_pred_1"]
            fp = metrics[f"{split}_actual_0_pred_1"]
            tn = metrics[f"{split}_actual_0_pred_0"]
            fn = metrics[f"{split}_actual_1_pred_0"]
            p = tp + fn
            n = tn + fp
            tpr = tp / p
            fpr = fp / n
            plr = tpr / fpr
            metrics[f"{split}_PositiveLikelihoodRatio"] = plr

    return metrics, roc_metrics


def make_prediction_test(
    model,
    test_loader,
    device,
    num_classes,
    explain=False,
):
    """Make predictions using the model

    Args:
        model (pytorch geo model) : Model that will make predictiions
        test_loader (torch dataloader): Dataloader for the
            test dataset
        device (gpu or cpu): Device to evaluate the model
            on
        num_classes (int) : Number of classes in the dataset
        explain (bool) : Whether this is for an explain dataset

    Returns:
        metrics (dict) : Dict containing the metrics"""

    model.to(device)

    test_metrics = MetricCollection(
        MulticlassConfusionMatrix(num_classes=num_classes),
        Recall(task="multiclass", num_classes=num_classes, average="none"),
        Precision(task="multiclass", num_classes=num_classes, average="none"),
        F1Score(task="multiclass", num_classes=num_classes, average="none"),
        MulticlassJaccardIndex(num_classes=num_classes, average="none"),
        Accuracy(task="multiclass", num_classes=num_classes, average="none"),
    ).to(device)

    test_metrics_roc = MetricCollection(
        ROC(task="multiclass", num_classes=num_classes),
        AUROC(task="multiclass", num_classes=num_classes, average="none"),
    ).to(device)

    # test data
    model.eval()
    for index, data in enumerate(test_loader):
        with torch.no_grad():
            # note set to none is meant to have less memory footprint
            # move data to device
            data.to(device)

            # forward pass - with autocasting
            with torch.autocast(device_type="cuda"):
                if explain:
                    warnings.warn(
                        "Assuming that last layer is unnormalised\
                                  prob"
                    )
                    output = model(data.x, data.edge_index, data.batch)
                    output = output.log_softmax(dim=-1)
                    print("output", output)
                    print("label", data.y)
                else:
                    output = model(data)

                # output is log softmax therefore convert back to prob
                test_metrics_roc.update(torch.exp(output), data.y)

                # argmax predictions
                test_predictions = output.argmax(dim=1)
                test_metrics.update(test_predictions, data.y)

    # metric over all batches
    test_metrics = test_metrics.compute()
    test_metrics_roc = test_metrics_roc.compute()

    # output metrics
    metrics = {}
    roc_metrics = {}

    # make into format acceptable to wandb
    for i in range(num_classes):
        metrics[f"TestRecall_{i}"] = test_metrics["MulticlassRecall"][i]
        metrics[f"TestPrecision_{i}"] = test_metrics["MulticlassPrecision"][i]
        metrics[f"TestF1Score_{i}"] = test_metrics["MulticlassF1Score"][i]
        metrics[f"TestJaccardIndex_{i}"] = test_metrics["MulticlassJaccardIndex"][i]
        metrics[f"TestAccuracy_{i}"] = test_metrics["MulticlassAccuracy"][i]
        for j in range(num_classes):
            metrics[f"Test_actual_{i}_pred_{j}"] = test_metrics[
                "MulticlassConfusionMatrix"
            ][i][j]

        roc_metrics[f"TestFPR_{i}"] = test_metrics_roc["MulticlassROC"][0][i]
        roc_metrics[f"TestTPR_{i}"] = test_metrics_roc["MulticlassROC"][1][i]
        roc_metrics[f"TestThreshold_{i}"] = test_metrics_roc["MulticlassROC"][2][i]
        metrics[f"TestAUROC_{i}"] = test_metrics_roc["MulticlassAUROC"][i]

    if num_classes == 2:
        # ISSUE: This metric is specific only to when have two classes
        tp = metrics["Test_actual_1_pred_1"]
        fp = metrics["Test_actual_0_pred_1"]
        tn = metrics["Test_actual_0_pred_0"]
        fn = metrics["Test_actual_1_pred_0"]
        p = tp + fn
        n = tn + fp
        tpr = tp / p
        fpr = fp / n
        plr = tpr / fpr
        metrics["Test_PositiveLikelihoodRatio"] = plr

    return metrics, roc_metrics
