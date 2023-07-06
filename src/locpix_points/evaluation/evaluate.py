"""Evaluation module

This contains functions for evaluating the models
"""

import torch
from torchmetrics.classification import BinaryAccuracy

def make_prediction(model, optimiser, train_loader, val_loader, device, label_level):
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
        label_level (string) : Either node or graph"""
        
    model.to(device)

    
    train_accuracy = BinaryAccuracy().to(device)
    val_accuracy = BinaryAccuracy().to(device)

    # training data
    model.eval()
    for index, data in enumerate(train_loader):
        with torch.no_grad():

            # note set to none is meant to have less memory footprint
            optimiser.zero_grad(set_to_none=True)

            # move data to device
            data.to(device)

            # forward pass - with autocasting
            with torch.autocast(device_type='cuda'):
                output = model(data)
                train_predictions = output.argmax(dim=1)

                # per batch metric
                train_accuracy.update(train_predictions, data.y)

    # metric over all batches
    train_acc = train_accuracy.compute()
            
    for index, data in enumerate(val_loader):
        with torch.no_grad():

            # note set to none is meant to have less memory footprint
            optimiser.zero_grad(set_to_none=True)

            # move data to device
            data.to(device)

            # forward pass - with autocasting
            with torch.autocast(device_type='cuda'):
                output = model(data)
                val_predictions = output.argmax(dim=1)

                # per batch metric
                val_accuracy.update(val_predictions, data.y)

    # metric over all batches
    val_acc = val_accuracy.compute()

    return train_acc, val_acc
