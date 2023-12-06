"""Training module.

This module contains definitions relevant to
training the model.

"""

import torch

import wandb


def train_loop(
    epochs,
    model,
    optimiser,
    train_loader,
    val_loader,
    loss_fn,
    device,
    label_level,
    num_train_graph,
    num_val_graph,
    model_path,
):
    """This is the main function which defines
    a training loop.

    Args:
        epochs (int): Number of epochs to train for
        optimiser (torch.optim optimiser): Optimiser
            that controls training of the model
        model (torch geometric model): Model that
            is going to be trained
        train_loader (torch dataloader): Dataloader for the
            training dataset
        val_loader (torch dataloader): Dataloader for the
            validation dataset
        loss_fn (loss function): Loss function calculate loss
            between train and output
        device (gpu or cpu): Device to train the model
            on
        label_level (string) : Either node or graph
        num_train_graph (int) : Number of graphs in train set
        num_val_graph (int) : Number of graphs in val set
        model_path (string) : Where to save the model to"""

    model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    best_loss = 1e10

    for epoch in range(epochs):
        print("Epoch: ", epoch)

        # TODO : autocast - look at gradient accumulation and take care with multiple gpus

        running_train_loss = 0
        num_train_node = 0
        running_val_loss = 0
        num_val_node = 0

        # training data
        model.train()
        for index, data in enumerate(train_loader):
            # note set to none is meant to have less memory footprint
            optimiser.zero_grad(set_to_none=True)

            # move data to device
            data.to(device)

            # forward pass - with autocasting
            with torch.autocast(device_type="cuda"):
                output = model(data)
                loss = loss_fn(output, data.y)
                running_train_loss += loss

            # scales loss - calls backward on scaled loss creating scaled gradients
            scaler.scale(loss).backward()

            # metrics
            num_train_node += data.num_nodes

            # unscales the gradients of optimiser then optimiser.step is called
            scaler.step(optimiser)

            # update scale for next iteration
            scaler.update()

        # val data
        # TODO: make sure torch.no_grad() somewhere
        # make sure model in eval mode
        model.eval()
        for index, data in enumerate(val_loader):
            with torch.no_grad():
                # note set to none is meant to have less memory footprint
                optimiser.zero_grad(set_to_none=True)

                # move data to device
                data.to(device)

                # forward pass - with autocasting
                with torch.autocast(device_type="cuda"):
                    output = model(data)
                    loss = loss_fn(output, data.y)
                    running_val_loss += loss
                    num_val_node += data.num_nodes

        # divide by number of graphs as we reduce the loss by the mean already
        # or number of locs
        running_train_loss /= num_train_graph
        running_val_loss /= num_val_graph
        # elif label_level == "node":
        #    running_train_loss /= num_train_node
        #    running_val_loss /= num_val_node

        # log results
        print("Train loss", running_train_loss)
        print("Val loss", running_val_loss)
        wandb.log({"train_loss": running_train_loss, "val_loss": running_val_loss})

        # if loss lowest on validation set save it
        if running_val_loss < best_loss:
            best_loss = running_val_loss
            print("Saving model new lowest loss on val set")
            torch.save(model.state_dict(), model_path)

    print("Number of train nodes", num_train_node)
    print("Number of val nodes", num_val_node)
