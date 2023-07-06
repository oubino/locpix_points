"""Train recipe

Recipe :
    1. Initialise dataset
    2. Initialise dataloader
    3. Train...
"""

import os
import yaml
from locpix_points.data_loading import datastruc
import torch_geometric.loader as L
from locpix_points.training import train
from locpix_points.models import model_choice
from locpix_points.evaluation import evaluate
from torchsummary import summary
import torch.optim
import argparse
import time
# import torch
# import torch_geometric.transforms as T


def main():

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Training"
    )

    parser.add_argument(
        "-i", 
        "--project_directory", 
        action="store",
        type=str, 
        help="location of the project directory", 
        required=True,
    )
    
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for processing",
        required=True,
    )

    args = parser.parse_args()

    project_directory = args.project_directory

    # load yaml
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in config
    batch_size = config['batch_size']
    epochs = config['epochs']
    gpu = config['gpu']
    optimiser = config['optimiser']
    lr = config['lr']
    weight_decay = config['weight_decay']
    num_workers = config['num_workers']
    loss_fn = config['loss_fn']
    label_level = config['label_level']

    # define device
    if gpu is True and not torch.cuda.is_available():
        raise ValueError('No gpu available, can run on cpu\
                         instead')
    elif gpu is True and torch.cuda.is_available():
        device = torch.device('cuda')
    elif gpu is False:
        device = torch.device('cpu')
    else:
        raise ValueError('Specify cpu or gpu !')

    # folder
    processed_directory = os.path.join(project_directory, "processed")
    train_folder = os.path.join(processed_directory, 'train')
    val_folder = os.path.join(processed_directory, 'val')
    test_folder = os.path.join(processed_directory, 'test')

    # transform
    # TODO: #4 add in transforms, and ensure specified in config file
    train_transform = None
    val_transform = None
    # Transforms that appear to be good
    # normalize rotation
    # normalizes scale
    # random jitter
    # random flip
    # random rotate
    # random shear
    # normalize features
    # knngraph
    # radius graph
    # gdc
    # gcnnorm
    # feature propagation
    # e.g. T.compose([T.ToUndirected(), T.AddSelfLoops()])

    # load in train dataset
    train_set = datastruc.SMLMDataset(None,
                                      None,
                                      train_folder,
                                      transform=train_transform,
                                      pre_transform=None,
                                      pre_filter=None,
                                      gpu=gpu)

    # load in val dataset
    val_set = datastruc.SMLMDataset(None,
                                    None,
                                    val_folder,
                                    transform=val_transform,
                                    pre_transform=None,
                                    pre_filter=None,
                                    gpu=gpu)

    # TODO: #5 configuration for dataloaders

    # if data is on gpu then don't need to pin memory
    # and this causes errors if try
    if gpu is True:
        pin_memory = False
    elif gpu is False:
        pin_memory = True
    else:
        raise ValueError('gpu should be True or False')

    # initialise dataloaders
    train_loader = L.DataLoader(train_set, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=pin_memory,
                                num_workers=num_workers)
    val_loader = L.DataLoader(val_set,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=pin_memory,
                              num_workers=num_workers)

    # initialise model
    model = model_choice(config['model'],
                         # this should parameterise the chosen model
                         config[config['model']])

    # initialise optimiser
    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(),
                                     lr = lr,
                                     weight_decay = weight_decay)
    
    # initialise loss function
    if loss_fn == 'nll':
        loss_fn = torch.nn.functional.nll_loss

    # print parameters
    print('\n')
    print('---- Params -----')
    print('\n')
    print('Input features: ', train_set.num_node_features)
    print('Num classes: ', train_set.num_classes)
    print('Batch size: ', batch_size)
    print('Epochs: ', epochs)
    num_train_graph = len(train_set)
    print('Number train graphs', num_train_graph)
    num_val_graph = len(val_set)
    print('Number val graphs', num_val_graph)
    first_train_item = train_set.get(0)
    nodes = first_train_item.num_nodes
    label = first_train_item.y
    if label_level == "node":
        assert label.shape[0] == nodes
    elif label_level == "graph":
        assert label.shape == 1
    else:
        raise ValueError("Label level not defined")

    # model summary
    print('\n')
    print('---- Model summary (estimate) ----')
    print('\n')
    number_nodes = nodes*len(train_set)  # this is just for summary, has no bearing on training
    summary(model, input_size=(train_set.num_node_features, number_nodes),
            batch_size=batch_size)

    # train loop
    print('\n')
    print('---- Training... ----')
    print('\n')
    train.train_loop(epochs,
                     model,
                     optimiser,
                     train_loader,
                     val_loader,
                     loss_fn,
                     device,
                     label_level,
                     num_train_graph ,
                     num_val_graph,
                     lr,
                     wandb_project=config['wandb_project'],
                     wandb_dataset=config['wandb_dataset']
                     )
    print('\n')
    print('---- Finished training... ----')
    print('\n')

    # save final model
    print('\n')
    print('---- Saving final model... ----')
    model_folder = os.path.join(project_directory, "models")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_path = f"{config['wandb_project']}_{config['wandb_dataset']}_{time.asctime(time.gmtime(time.time()))}_.pt"
    model_path = os.path.join(model_folder, model_path)
    #input('stop', need to check weights have changed after training)
    torch.save(model.state_dict(), model_path)
    print('\n')

    print('\n')
    print('---- Predict on train & val set... ----')
    print('\n')
    train_acc, val_acc = evaluate.make_prediction(model,
                                                    optimiser,
                                                    train_loader,
                                                    val_loader,
                                                    device,
                                                    label_level)
    print('Train accuracy: ', train_acc)
    print('Validation accuracy: ', val_acc)

    # save config file
    yaml_save_loc = os.path.join(
        project_directory, 'train.yaml'
    )
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)

if __name__ == "__main__":
    main()
         
