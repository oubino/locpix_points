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
import json
import warnings


class SMLMDataset(Dataset):
    """Base SMLM dataset class.

    All the SMLM dataitems are defined in this class.
    Process is defined in the derived class

    Attributes:
        raw_loc_dir_root (string): Directory of the the folder
            which contains the "raw" localisation dataset i.e. the parquet files,
            is not technically raw as has passed through
            our preprocessing module - bear this in mind
        raw_cluster_dir_root (string): Directory of the the folder
            which contains the "raw" cluster dataset i.e. the parquet files,
            is not technically raw as has passed through
            our preprocessing module - bear this in mind
        processed_dir_root (string): Directory of the the folder
            which contains the the directory of the
            processed dataset - processed via pygeometric
            i.e. Raw .csv/.parquet -> Preprocessing module outputs to
            raw_dir -> Taken in to data_loading module processed
            to processed_dir -> Then pytorch analysis begins
        label_level (string) : Either "graph" or "node". In the former
            label labels the whole graph in later label per node.
            There will often only be one option but this is in case you
            have graph label and node label
        pre_filter (function) : Takes in data object and returns 1 if
            data should be included in dataset and 0 if it should not
        gpu (boolean): Whether the data should be savedd from the GPU
            or not.
        transform (dict) : Transforms to be applied to each data point.
            Keys are the transforms and values are the relevant
            parameters if applicable
        _data_list (list): Data from the dataset
            so can access via a numerical index later.
    """

    def __init__(
        self,
        raw_loc_dir_root,
        raw_cluster_dir_root,
        processed_dir_root,
        label_level,
        pre_filter,
        gpu,
        transform,
        pre_transform,
    ):
        # index the dataitems (idx)
        self._raw_loc_dir_root = raw_loc_dir_root
        self._raw_cluster_dir_root = raw_cluster_dir_root
        self._processed_dir_root = processed_dir_root
        if self._raw_loc_dir_root is not None:
            self._raw_loc_file_names = list(sorted(os.listdir(raw_loc_dir_root)))
        if self._raw_cluster_dir_root is not None:
            self._raw_cluster_file_names = list(
                sorted(os.listdir(raw_cluster_dir_root))
            )
            if self._raw_loc_dir_root is not None:
                assert self._raw_cluster_file_names == self._raw_loc_file_names
        self._processed_file_names = list(sorted(os.listdir(processed_dir_root)))
        self.label_level = label_level
        self.gpu = gpu

        if transform is None or len(transform) == 0:
            super().__init__(None, None, pre_transform, pre_filter)

        else:
            # define transforms
            output_transforms = []

            # axis to rotate around i.e. axis=2 rotate around z axis - meaning
            # coordinates are rotated in the xy plane
            if "z_rotate" in transform.keys():
                output_transforms.append(transforms.RandomRotate(degrees=180, axis=2))

            # need to either define as constant or allow precision to impact this
            if "jitter" in transform.keys():
                output_transforms.append(transforms.RandomJitter(transform["jitter"]))

            # axis = 0 - means x coordinates are flipped - i.e. reflection
            # in the y axis
            if "x_flip" in transform.keys():
                output_transforms.append(transforms.RandomFlip(axis=0))

            # axis = 1 - means y coordinates are flipped - i.e. reflection
            # in the x axis
            if "y_flip" in transform.keys():
                output_transforms.append(transforms.RandomFlip(axis=1))

            # need to define scale factor interval in config
            if "randscale" in transform.keys():
                output_transforms.append(
                    transforms.RandomScale(scales=tuple(transform["randscale"]))
                )

            # shear by particular matrix
            if "shear" in transform.keys():
                output_transforms.append(transforms.RandomShear(transform["shear"]))

            if "subsample" in transform.keys():
                output_transforms.append(
                    custom_transforms.Subsample(
                        transform["subsample"][0], transform["subsample"][1]
                    )
                )

            if "normalisescale" in transform.keys():
                raise ValueError('Normalise and sacle the data myself to bewteen -1 nad 1 or 0 and 1; dont use this + make sure clusters scaled correctly')
                output_transforms.append(transforms.NormalizeScale())

            output_transforms = transforms.Compose(output_transforms)

            super().__init__(None, output_transforms, pre_transform, pre_filter)

    @property
    def has_download(self) -> bool:
        return False

    @property
    def raw_loc_dir(self) -> str:
        return self._raw_loc_dir_root

    @property
    def raw_cluster_dir(self) -> str:
        return self._raw_cluster_dir_root

    @property
    def raw_loc_file_names(self):
        return self._raw_loc_file_names

    @property
    def raw_cluster_file_names(self):
        return self._raw_cluster_file_names

    @property
    def processed_dir(self) -> str:
        return self._processed_dir_root

    @property
    def processed_file_names(self):
        return self._processed_file_names

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


class LocDataset(SMLMDataset):
    """Dataset for localisations with no clusters loaded in"""

    def __init__(
        self,
        raw_loc_dir_root,
        raw_cluster_dir_root,
        processed_dir_root,
        label_level,
        pre_filter,
        gpu,
        transform,
        pre_transform,
        min_feat,
        max_feat,
        pos,
        feat,
    ):
        super().__init__(
            raw_loc_dir_root,
            raw_cluster_dir_root,
            processed_dir_root,
            label_level,
            pre_filter,
            gpu,
            transform,
            pre_transform,
        )

    def process(self):
        raise NotImplementedError


class ClusterDataset(SMLMDataset):
    """Dataset for clusters"""

    def __init__(
        self,
        raw_loc_dir_root,
        raw_cluster_dir_root,
        processed_dir_root,
        label_level,
        pre_filter,
        gpu,
        transform,
        pre_transform,
        min_feat,
        max_feat,
        pos,
        feat,
    ):
        super().__init__(
            raw_loc_dir_root,
            raw_cluster_dir_root,
            processed_dir_root,
            label_level,
            pre_filter,
            gpu,
            transform,
            pre_transform,
        )

    def process(self):
        raise NotImplementedError


class ClusterLocDataset(SMLMDataset):
    """Dataset for localisations with localisations connected within clusters.
    Clusters connected to nearest k neighbours.

    Args:
        dataset_type (str) : Name identifying the type of dataset
        loc_feat (list) : List of features to consider in localisation dataset
        cluster_feat (list) : List of features to consider in cluster dataset
        min_feat_locs (dict) : Minimum values of features for the locs training dataset
        max_feat_locs (dict) : Maxmimum values of features over locs training dataset
        min_feat_clusters (dict) : Minimum values of features for the clusters training dataset
        max_feat_clusters (dict) : Maxmimum values of features over clusters training dataset
        kneighbours (int) : Number of neighbours each cluster connected to
    """

    def __init__(
        self,
        raw_loc_dir_root,
        raw_cluster_dir_root,
        processed_dir_root,
        label_level,
        pre_filter,
        gpu,
        transform,
        pre_transform,
        loc_feat,
        cluster_feat,
        min_feat_locs,
        max_feat_locs,
        min_feat_clusters,
        max_feat_clusters,
        kneighbours,
    ):
        self.dataset_type = "ClusterLocDataset"
        self.loc_feat = loc_feat
        self.cluster_feat = cluster_feat
        self.min_feat_locs = min_feat_locs
        self.max_feat_locs = max_feat_locs
        self.min_feat_clusters = min_feat_clusters
        self.max_feat_clusters = max_feat_clusters
        self.kneighbours = kneighbours

        super().__init__(
            raw_loc_dir_root,
            raw_cluster_dir_root,
            processed_dir_root,
            label_level,
            pre_filter,
            gpu,
            transform,
            pre_transform,
        )

    def process(self):
        """Process the raw data into heterogeneous graph"""

        idx = 0
        idx_to_name = {"idx": [], "file_name": []}

        # convert raw parquet files to tensors
        # note that cluster and loc file names should be the same therefore choosing one should
        # have no impact
        for raw_path in self.raw_loc_file_names:
            # load in and process localisation data
            loc_path = os.path.join(self._raw_loc_dir_root, raw_path)
            loc_table = pq.read_table(loc_path)

            # metadata load in
            dimensions = loc_table.schema.metadata[b"dim"]
            dimensions = int(dimensions)

            gt_label_scope = loc_table.schema.metadata[b"gt_label_scope"].decode(
                "utf-8"
            )
            if gt_label_scope == "loc":
                if self.label_level != "node":
                    raise ValueError(
                        "You cannot specify graph level label when the gt label is per loc/node. Amend process configuration file"
                    )
            elif gt_label_scope == "fov":
                if self.label_level != "graph":
                    raise ValueError(
                        "You cannot specify node level label when dataset has per fov/graph labels. Amend process config file"
                    )
            else:
                raise ValueError("No gt label scope")

            gt_label_map = json.loads(
                loc_table.schema.metadata[b"gt_label_map"].decode("utf-8")
            )
            gt_label_map = {int(key): value for key, value in gt_label_map.items()}
            self.gt_label_map = gt_label_map

            # load in and process cluster data
            cluster_path = os.path.join(self._raw_cluster_dir_root, raw_path)
            cluster_table = pq.read_table(cluster_path)

            # each dataitem is a homogeneous graph
            data = HeteroData()

            # load position (if present) and features to data
            data = features.load_loc_cluster(
                data,
                loc_table,
                cluster_table,
                self.loc_feat,
                self.cluster_feat,
                self.min_feat_locs,
                self.max_feat_locs,
                self.min_feat_clusters,
                self.max_feat_clusters,
                self.kneighbours,
            )

            # load in gt label
            gt_label = loc_table.schema.metadata[b"gt_label"]
            gt_label = int(gt_label)

            # load gt label to data
            if self.label_level == "graph":
                if gt_label is None:
                    raise ValueError("No GT label for the FOV")
                if "gt_label" in loc_table.columns:
                    raise ValueError("Should be no gt label column")
                else:
                    print("gt label", gt_label)
                    data.y = torch.tensor([gt_label], dtype=torch.long)
            elif self.label_level == "node":
                raise NotImplementedError()

            # assign name to data
            name = loc_table.schema.metadata[b"name"]
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
            torch.save(data, os.path.join(self.processed_dir, f"{idx}.pt"))

            # add to index
            idx_to_name["idx"].append(idx)
            idx_to_name["file_name"].append(file_name)
            idx += 1

        warnings.warn("Need to check values are correct for data, positions, features")
        warnings.warn("Check graph correctly connected")
        warnings.warn(
            "Consider what else may want to save for each dataitem: name of each feature? gt label map? scope? name? a lot of this is in the config files so would become redundant"
        )
        # save mapping from idx to name
        df = pl.from_dict(idx_to_name)
        df.write_csv(os.path.join(self.processed_dir, "file_map.csv"))


"""
def process_heterogeneous(self):
        Process the raw data into procesed data.
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
            3. Then the graph is saved

        idx = 0
        idx_to_name = {}

        raise ValueError("Haven't looked at in a while may need amending")

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

"""
