"""Datastruc module.

This module contains definitions of the datastructures the
SMLM dataitem will be parsed as during processing.
"""

import os
import torch
from torch_geometric.data import Dataset, HeteroData, Data
from torch_geometric import transforms
import pyarrow.parquet as pq
import pyarrow.compute as pc
import ast
import polars as pl
from . import features
from . import custom_transforms


class SMLMDataset(Dataset):
    """SMLM dataset class.

    All the SMLM dataitems are defined in this class.
    Assumption that name is the last part of the file
    name before the .file_extension.

    Attributes:
        heterogeneous (bool): If True then separate graph per channel
            i.e. heterogeneous data.
            If False then one graph for all data i.e. homogeneous
            graph
        raw_dir_root: A string with the directory of the the folder
            which contains the "raw" dataset i.e. the parquet files,
            is not technically raw as has passed through
            our preprocessing module - bear this in mind
        processed_dir_root: A string with the directory of the the folder
            which contains the the directory of the
            processed dataset - processed via pygeometric
            i.e. Raw .csv/.parquet -> Preprocessing module outputs to
            raw_dir -> Taken in to data_loading module processed
            to processed_dir -> Then pytorch analysis begins
        transform: The transform to be applied to each
                   loaded in graph/point cloud.
        pos (string) : How to load in the position
        feat (string) : How to load in the feature
        label_level (string) : Either "graph" or "node". In the former
            label labels the whole graph in later label per node.
            There will often only be one option but this is in case you
            have graph label and node label
        _data_list: A list of the data from the dataset
            so can access via a numerical index later.
    """

    def __init__(
        self,
        heterogeneous,
        raw_dir_root,
        processed_dir_root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        pos=None,
        feat=None,
        label_level=None,
        gpu=True,
        min_feat=None,
        max_feat=None,
    ):
        """Inits SMLMDataset with root directory where
        data is located and the transform to be applied when
        getting item.
        Note the pre_filter (non callable) is boolean? whether
        there is a pre-filter
        pre_filter (function) : Takes in data object and returns 1 if
            data should be included in dataset and 0 if it should not
        gpu (boolean): Whether the data should be savedd from the GPU
            or not.
        transform (dict) : Keys are the transforms and values are the relevant
            parameters if applicable
        min_feat (dict) : Minimum values for the features over the training dataset
        max_feat (dict) : Maximum values for the features over the training dataset"""

        self.heterogeneous = heterogeneous
        # index the dataitems (idx)
        self._raw_dir_root = raw_dir_root
        self._processed_dir_root = processed_dir_root
        self._raw_file_names = list(sorted(os.listdir(raw_dir_root)))
        self._processed_file_names = list(sorted(os.listdir(processed_dir_root)))
        self.gpu = gpu
        # Note deliberately set root to None
        # as going to overload the raw and processed
        # dir. This could cause problems so be aware
        self.pos = pos
        self.feat = feat
        self.label_level = label_level
        self.min_feat = min_feat
        self.max_feat = max_feat

        if transform is None or len(transform) == 0:
            super().__init__(None, None, pre_transform, pre_filter)

        else:

            # define transforms
            output_transforms = []
            
            # axis to rotate around i.e. axis=2 rotate around z axis - meaning
            # coordinates are rotated in the xy plane
            if 'z_rotate' in transform.keys():
                output_transforms.append(transforms.RandomRotate(degrees=180, axis=2))

            # need to either define as constant or allow precision to impact this
            if 'jitter' in transform.keys():
                output_transforms.append(transforms.RandomJitter(transform['jitter']))

            # axis = 0 - means x coordinates are flipped - i.e. reflection
            # in the y axis
            if 'x_flip' in transform.keys():
                output_transforms.append(transforms.RandomFlip(axis=0))

            # axis = 1 - means y coordinates are flipped - i.e. reflection
            # in the x axis 
            if 'y_flip' in transform.keys():
                output_transforms.append(transforms.RandomFlip(axis=1))

            # need to define scale factor interval in config
            if 'randscale' in transform.keys():
                output_transforms.append(transforms.RandomScale(scales=tuple(transform['randscale'])))

            # shear by particular matrix
            if 'shear' in transform.keys():
                output_transforms.append(transforms.RandomShear(transform['shear']))

            if 'subsample' in transform.keys():
                output_transforms.append(custom_transforms.Subsample(transform['subsample'][0], transform['subsample'][1]))

            if 'normalisescale' in transform.keys():
                output_transforms.append(transforms.NormalizeScale())

            output_transforms = transforms.Compose(output_transforms)

            super().__init__(None, output_transforms, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir_root

    @property
    def processed_dir(self) -> str:
        return self._processed_dir_root

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def process(self):
        if self.heterogeneous:
            self.process_heterogeneous()
        elif self.heterogeneous is False:
            self.process_homogeneous()

    def process_heterogeneous(self):
        """Process the raw data into procesed data.
        This currently includes
            1. For each .parquet create a heterogeneous graph
            , where the different (i.e. heterogeneous) nodes
            are due to there being multiple channels.
            e.g. two channel image with 700 localisations for
            channel 0 and 300 for channel 1 - would have
            1000 nodes and each node is type (channel 0 or
            channel 1)
            2. Then if not pre-filtered the heterogeneous
            graph is pre-transformed
            3. Then the graph is saved"""

        idx = 0
        idx_to_name = {}

        # convert raw parquet files to tensors
        for raw_path in self.raw_paths:
            # read in parquet file
            arrow_table = pq.read_table(raw_path)
            # dimensions and channels metadata
            dimensions = arrow_table.schema.metadata[b"dim"]
            channels = arrow_table.schema.metadata[b"channels"]
            dimensions = int(dimensions)
            channels = ast.literal_eval(channels.decode("utf-8"))
            # each dataitem is a heterogeneous graph
            # where the channels define the different type of node
            # i.e. for two channel data have two types of node
            # for both channels
            data = HeteroData()
            # for channel in list of channels
            for chan in channels:
                # filter table
                filter = pc.field("channel") == chan
                filter_table = arrow_table.filter(filter)
                # convert to tensor (Number of points x 2/3 (dimensions))
                x = torch.tensor(filter_table["x"].to_numpy())
                y = torch.tensor(filter_table["y"].to_numpy())
                if dimensions == 2:
                    coord_data = torch.stack((x, y), dim=1)
                if dimensions == 3:
                    z = torch.tensor(arrow_table["z"].to_numpy())
                    coord_data = torch.stack((x, y, z), dim=1)

                # feature tensor
                # shape: [Number of points x 2/3 dimensions]
                data[str(chan)].x = coord_data

                # position tensor
                # shape: [Number of points x 2/3 dimensions]
                data[str(chan)].pos = coord_data

                # localisation level labels
                data[str(chan)].y = torch.tensor(filter_table["gt_label"].to_numpy())

            _, extension = os.path.splitext(raw_path)
            _, tail = os.path.split(raw_path)
            file_name = tail.strip(extension)

            # assign name to data
            print("file name", file_name)
            name = arrow_table.schema.metadata[b"name"]
            print("name", name)
            input("stop dataloading")

            # if pre filter skips it then skip this item
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            # pre-transform
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # save it
            _, extension = os.path.splitext(raw_path)
            _, tail = os.path.split(raw_path)
            file_name = tail.strip(extension)
            # TODO: change/check this
            # if self.gpu:
            #    data.cuda()
            if data.x is not None:
                data.x = data.x.float()
            if data.pos is not None:
                data.pos = data.pos.float()
            if data.y is not None:
                data.y = data.y.long()
            torch.save(data, os.path.join(self.processed_dir, f"{idx}.pt"))

            # add to index
            idx_to_name["idx"].append(idx)
            idx_to_name["file_name"].append(file_name)
            idx += 1

        # save mapping from idx to name
        df = pl.from_dict(idx_to_name)
        df.write_csv(os.path.join(self.processed_dir, "file_map.csv"))

    def process_homogeneous(self):
        """Process the raw data into procesed data.
        This currently includes
            1. For each .parquet create a homogeneous graph
            2. Then if not pre-filtered the
            graph is pre-transformed
            3. Then the graph is saved"""

        idx = 0
        idx_to_name = {"idx": [], "file_name": []}

        # convert raw parquet files to tensors
        for raw_path in self.raw_paths:
            # read in parquet file
            arrow_table = pq.read_table(raw_path)
            # dimensions metadata
            dimensions = arrow_table.schema.metadata[b"dim"]
            dimensions = int(dimensions)
            # each dataitem is a homogeneous graph
            data = Data()

            # load position (if present) and features to data
            data = features.load_pos_feat(
                arrow_table, data, self.pos, self.feat, self.min_feat, self.max_feat
            )

            gt_label_fov = arrow_table.schema.metadata[b"gt_label_fov"]

            # load gt label to data
            if self.label_level == "graph":
                if gt_label_fov is None:
                    raise ValueError("No gt label for the fov")
                else:
                    data.y = gt_label_fov
            elif self.label_level == "node":
                data.y = torch.tensor(arrow_table["gt_label"].to_numpy())
            else:
                raise ValueError("Label level should be graph or node")

            # assign name to data
            name = arrow_table.schema.metadata[b"name"]
            name = str(name.decode("utf-8"))
            data.name = name

            # if pre filter skips it then skip this item
            # if pre_filter is 0 - data should not be included
            # and the if statement will read
            # if not 0
            # this is True and so continue will occur - i.e. data is skipped
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            # pre-transform
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # save it
            _, extension = os.path.splitext(raw_path)
            _, tail = os.path.split(raw_path)
            file_name = tail.strip(extension)
            # TODO: change/check this and make option in process
            # if self.gpu:
            #    data.cuda()
            if data.x is not None:
                data.x = data.x.float()
            if data.pos is not None:
                data.pos = data.pos.float()
            if data.y is not None:
                data.y = data.y.long()
            torch.save(data, os.path.join(self.processed_dir, f"{idx}.pt"))

            # add to index
            idx_to_name["idx"].append(idx)
            idx_to_name["file_name"].append(file_name)
            idx += 1

        # save mapping from idx to name
        df = pl.from_dict(idx_to_name)
        df.write_csv(os.path.join(self.processed_dir, "file_map.csv"))

    def len(self):
        files = self._processed_file_names
        if "pre_filter.pt" in files:
            files.remove("pre_filter.pt")
        if "pre_transform.pt" in files:
            files.remove("pre_transform.pt")
        if "file_map.csv" in files:
            files.remove("file_map.csv")
        return len(files)

    def get(self, idx):
        """I believe that pytorch geometric is wrapper
        over get item and therefore it handles the
        transform"""
        data = torch.load(os.path.join(self.processed_dir, f"{idx}.pt"))
        return data

    # This is copied from the pytorch geometric docs
    # because is not defined in my download for some reason
    def _infer_num_classes(self, y) -> int:
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return y.size(-1)

    # This is copied from the pytorch geometric docs
    # because is not defined in my download for some reason
    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        y = torch.cat([data.y for data in self], dim=0)
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, "_data_list") and self._data_list is not None:
            self._data_list = self.len() * [None]
        return self._infer_num_classes(y)
