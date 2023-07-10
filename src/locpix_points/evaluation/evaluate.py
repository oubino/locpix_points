"""Evaluation module

This contains functions for evaluating the models
"""

import torch
from torchmetrics import MetricCollection, Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import BinaryAccuracy, MulticlassConfusionMatrix, MulticlassJaccardIndex


def make_prediction(model, optimiser, train_loader, val_loader, device, label_level, num_classes):
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
        label_level (string) : Either node or graph
        num_classes (int) : Number of classes in the dataset"""

    model.to(device)

    train_metrics = MetricCollection(
        MulticlassConfusionMatrix(num_classes=num_classes),
        Recall(task="multiclass", num_classes=num_classes, average='none'),
        Precision(task="multiclass", num_classes=num_classes, average='none'),
        F1Score(task="multiclass", num_classes=num_classes, average='none'),
        MulticlassJaccardIndex(num_classses=num_classes, average='none'),
        Accuracy(task="multiclass", num_classes=num_classes, average='none')
    ).to(device)

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
                train_predictions = output.argmax(dim=1)

                # per batch metric
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
                val_predictions = output.argmax(dim=1)

                # per batch metric
                val_metrics.update(val_predictions, data.y)

    # metric over all batches
    train_metrics = train_metrics.compute()
    val_metrics = val_metrics.compute()

    print(train_metrics)
    print(val_metrics)

    input('stop')

    return train_metrics, val_metrics
