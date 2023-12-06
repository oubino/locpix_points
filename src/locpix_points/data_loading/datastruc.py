"""Datastruc module.

This module contains definitions of the datastructures the
SMLM dataitem will be parsed as during processing.
"""

import json
import os
import shutil
import warnings

import polars as pl
import pyarrow.parquet as pq
import torch
from torch_geometric import transforms
from torch_geometric.data import Data, Dataset, HeteroData

from . import custom_transforms, features


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
        save_on_gpu (boolean): Whether the data should be savedd to the GPU
            or not.
        transform (dict) : Transforms to be applied to each data point.
            Keys are the transforms and values are the relevant
            parameters if applicable
        _data_list (list): Data from the dataset
            so can access via a numerical index later.
        fov_x (float) : Size of fov in units for data (x)
        fov_y (float) : Size of fov in units for data (y)
    """

    def __init__(
        self,
        raw_loc_dir_root,
        raw_cluster_dir_root,
        processed_dir_root,
        label_level,
        pre_filter,
        save_on_gpu,
        transform,
        pre_transform,
        fov_x,
        fov_y,
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
        self.save_on_gpu = save_on_gpu
        self.fov_x = fov_x
        self.fov_y = fov_y

        if transform is None or len(transform) == 0:
            super().__init__(None, None, pre_transform, pre_filter)

        else:
            # define transforms
            output_transforms = []

            # axis to rotate around i.e. axis=2 rotate around z axis - meaning
            # coordinates are rotated in the xy plane
            if "z_rotate" in transform.keys():
                output_transforms.append(
                    custom_transforms.RandomRotate(degrees=180, axis=2)
                )

            # need to either define as constant or allow precision to impact this
            if "jitter" in transform.keys():
                output_transforms.append(
                    custom_transforms.RandomJitter(transform["jitter"])
                )

            # axis = 0 - means x coordinates are flipped - i.e. reflection
            # in the y axis
            if "x_flip" in transform.keys():
                output_transforms.append(custom_transforms.RandomFlip(axis=0))

            # axis = 1 - means y coordinates are flipped - i.e. reflection
            # in the x axis
            if "y_flip" in transform.keys():
                output_transforms.append(custom_transforms.RandomFlip(axis=1))

            # need to define scale factor interval in config
            if "randscale" in transform.keys():
                output_transforms.append(
                    custom_transforms.RandomScale(scales=tuple(transform["randscale"]))
                )

            # shear by particular matrix
            if "shear" in transform.keys():
                output_transforms.append(
                    custom_transforms.RandomShear(transform["shear"])
                )

            if "subsample" in transform.keys():
                raise ValueError("Not implemented for cluster/locs yet")
                output_transforms.append(
                    custom_transforms.Subsample(
                        transform["subsample"][0], transform["subsample"][1]
                    )
                )

            if "normalisescale" in transform.keys():
                raise ValueError(
                    "Normalise and sacle the data myself to bewteen -1 nad 1 or 0 and 1; dont use this + make sure clusters scaled correctly"
                )

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
        """Gets length of dataset

        Returns:
            len(files) (int) : Length of dataset"""
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
        transform

        Args:
            idx (int): Integer id for the data item

        Returns:
            data (torch tensor): Tensor data item from the dataset"""

        data = torch.load(os.path.join(self.processed_dir, f"{idx}.pt"))
        return data

    # This is copied from the pytorch geometric docs
    # because is not defined in my download for some reason
    def _infer_num_classes(self, y) -> int:
        """Get the number of classes

        Args:
            y (torch.tensor) : Label for the dataset

        Returns:
            num_classes (int) : Number of classes"""
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
        r"""Returns the number of classes in the dataset.

        Returns:
            self._infer_num_classes(y) (int) : Number of classes in the dataset"""
        y = torch.cat([data.y for data in self], dim=0)
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, "_data_list") and self._data_list is not None:
            self._data_list = self.len() * [None]
        return self._infer_num_classes(y)


class LocDataset(SMLMDataset):
    """Dataset for localisations with no clusters loaded in

    Raises:
        NotImplementedError: If try to process as this dataset has not been written yet
    """

    def __init__(
        self,
        raw_loc_dir_root,
        raw_cluster_dir_root,
        processed_dir_root,
        label_level,
        pre_filter,
        save_on_gpu,
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
            save_on_gpu,
            transform,
            pre_transform,
        )

    def process(self):
        raise NotImplementedError


class ClusterDataset(SMLMDataset):
    """Dataset for clusters"""

    def __init__(
        self,
        raw_cluster_dir_root,
        processed_dir_root,
        label_level,
        pre_filter,
        save_on_gpu,
        transform,
        pre_transform,
        fov_x,
        fov_y,
        from_cluster_loc=False,
        loc_net=None,
        device=torch.device("cpu"),
    ):
        self.from_cluster_loc = from_cluster_loc
        self.loc_net = loc_net
        self.device = device

        super().__init__(
            None,
            raw_cluster_dir_root,
            processed_dir_root,
            label_level,
            pre_filter,
            save_on_gpu,
            transform,
            pre_transform,
            fov_x,
            fov_y,
        )

    def process(self):
        """Processes data with just clusters"""

        # Processes data into homogeneous graph
        if self.from_cluster_loc:
            assert self.loc_net is not None

            # work through tensors
            for raw_path in self.raw_cluster_file_names:
                # if file not file_map.csv; pre_filter.pt; pre_transform.pt
                if raw_path in ["file_map.csv", "pre_filter.pt", "pre_transform.pt"]:
                    src = os.path.join(self._raw_cluster_dir_root, raw_path)
                    dst = os.path.join(self.processed_dir, raw_path)
                    shutil.copyfile(src, dst)
                    continue

                # initialise homogeneous data item
                data = Data()

                # load in tensor
                hetero_data = torch.load(
                    os.path.join(self._raw_cluster_dir_root, raw_path)
                )
                hetero_data.to(self.device)

                # pass through loc net
                x_dict, _, edge_index_dict = self.loc_net(hetero_data)
                data.x = x_dict["clusters"]
                data.edge_index = edge_index_dict["clusters", "near", "clusters"]

                # save to the homogeneous data item
                data.name = hetero_data.name
                data.y = hetero_data.y

                # save
                torch.save(data, os.path.join(self.processed_dir, raw_path))

            self._processed_file_names = list(sorted(os.listdir(self.processed_dir)))


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
        fov_x (float) : Size of fov in units for data (x)
        fov_y (float) : Size of fov in units for data (y)
    """

    def __init__(
        self,
        raw_loc_dir_root,
        raw_cluster_dir_root,
        processed_dir_root,
        label_level,
        pre_filter,
        save_on_gpu,
        transform,
        pre_transform,
        loc_feat,
        cluster_feat,
        min_feat_locs,
        max_feat_locs,
        min_feat_clusters,
        max_feat_clusters,
        kneighboursclusters,
        fov_x,
        fov_y,
        kneighbourslocs,
    ):
        self.dataset_type = "ClusterLocDataset"
        self.loc_feat = loc_feat
        self.cluster_feat = cluster_feat
        self.min_feat_locs = min_feat_locs
        self.max_feat_locs = max_feat_locs
        self.min_feat_clusters = min_feat_clusters
        self.max_feat_clusters = max_feat_clusters
        self.kneighboursclusters = kneighboursclusters
        self.kneighbourslocs = kneighbourslocs

        super().__init__(
            raw_loc_dir_root,
            raw_cluster_dir_root,
            processed_dir_root,
            label_level,
            pre_filter,
            save_on_gpu,
            transform,
            pre_transform,
            fov_x,
            fov_y,
        )

    def process(self):
        """Process the raw data into heterogeneous graph

        Raises:
            ValueError: If gt label is not in the correct format given the desired label
            NotImplementedError: If label level is node as currently not implemented"""

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
                self.kneighboursclusters,
                self.fov_x,
                self.fov_y,
                self.kneighbourslocs,
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
                    # print("gt label", gt_label)
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
            # if self.save_on_gpu:
            #    data.cuda()
            torch.save(data, os.path.join(self.processed_dir, f"{idx}.pt"))

            # add to index
            idx_to_name["idx"].append(idx)
            idx_to_name["file_name"].append(file_name)
            idx += 1

        self._processed_file_names = list(sorted(os.listdir(self.processed_dir)))

        warnings.warn("Need to check values are correct for data, positions, features")
        warnings.warn("Check graph correctly connected")
        warnings.warn(
            "Consider what else may want to save for each dataitem: name of each feature? gt label map? scope? name? a lot of this is in the config files so would become redundant"
        )
        # save mapping from idx to name
        df = pl.from_dict(idx_to_name)
        df.write_csv(os.path.join(self.processed_dir, "file_map.csv"))
