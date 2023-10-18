"""Preprocessing module.

This module contains functions to preprocess the data,
including (add as added):
- file to datastructure
    - convert .csv to datastructure
    - convert .parquet to datastructure
"""

import polars as pl
import os
from . import datastruc
import warnings
import pyarrow.parquet as pq
import json

def file_to_datastruc(
    input_file,
    dim,
    channel_col,
    frame_col,
    x_col,
    y_col,
    z_col,
    channel_choice,
    channel_label=None,
    gt_label_scope=None,
    gt_label_loc=None,
    features=None,
):
    """Loads in .csv or .parquet and converts to the required datastructure.

    Currently considers the following columns: channel frame x y z
    Also user can specify the channels they want to consider, these
    should be present in the channels column

    Args:
        input_file (string) : Location of the file
        save_loc (string) : Location to save datastructure to
        dim (int) : Dimensions to consider either 2 or 3
        channel_col (string) : Name of column which gives channel
            for localisation
        frame_col (string) : Name of column which gives frame for localisation
        x_col (string) : Name of column which gives x for localisation
        y_col (string) : Name of column which gives y for localisation
        z_col (string) : Name of column which gives z for localisation
        channel_choice (list of ints) : This will be list
            of integers representing channels to be considered
        channel_label (dictionary) : This is the
            label for each channel i.e. [0:'egfr',1:'ereg',2:'unk'] means
            channel 0 is egfr protein, channel 1 is ereg proteins and
            channel 2 is unknown
        gt_label_scope (string) : If not specified (None) there are no gt labels. If
            specified then is either 'loc' - gt label per localisatoin or 'fov' - gt
            label for field-of-view
        gt_label_loc (dict) : If specified then this contains a dictionary with two
            keys 'gt_label_col' giving the column the gt_label is in and 'gt_label_map'
            a dictionary with keys represetning the gt label present
            in the dataset and the value erepresenting the
            real concept e.g. {0:'dog', 1:'cat'}
        features (list) : List of features to consider for each localisation

    Returns:
        datastruc (SMLM_datastruc) : Datastructure containg the data
    """

    # Check dimensions correctly specified
    if dim != 2 and dim != 3:
        raise ValueError("Dimensions must be 2 or 3")
    if dim == 2 and z_col:
        raise ValueError("If dimensions are two no z should be specified")
    if dim == 3 and not z_col:
        raise ValueError("If dimensions are 3 then z_col must be specified")

    # first check all channels desired have specified label
    if channel_col is not None:
        assert sorted(list(channel_label.keys())) == channel_choice

    # which columns to be loaded in
    columns = [x_col, y_col]
    column_names = ["x", "y"]
    if dim == 3:
        columns.append(z_col)
        column_names.append("z")
    if frame_col is not None:
        columns.append(frame_col)
        column_names.append("frame")
    if channel_col is not None:
        columns.append(channel_col)
        column_names.append("channel")
    if len(features) != 0:
        columns.extend(features)
        column_names.extend(features)

    # check what gt labels have been loaded in if any
    if gt_label_scope == 'loc':
        assert gt_label_loc is not None
        gt_label_col = gt_label_loc['gt_label_col']
        gt_label_map = gt_label_loc['gt_label_map']
        gt_label = None
        columns.append(gt_label_col)
        column_names.append('gt_label')
        print('Per localisation labels')
        arrow_table = pq.read_table(input_file, columns=columns)
    elif gt_label_scope == 'fov':
        print('Per fov labels')
        arrow_table = pq.read_table(input_file, columns=columns)
        gt_label_map = json.loads(
            arrow_table.schema.metadata[b"gt_label_map"].decode("utf-8")
        )
        gt_label_map = {int(key): value for key, value in gt_label_map.items()}
        gt_label = arrow_table.schema.metadata[b"gt_label"]
        gt_label = int(gt_label)
        assert 'gt_label' not in arrow_table.columns
    elif gt_label_scope is None:
        print('No labels')
        arrow_table = pq.read_table(input_file, columns=columns)
        gt_label_map = None
        gt_label = None
    else:
        raise ValueError('gt_label_scope should be loc, fov or None')

    # Load in data
    arrow_table = pq.read_table(input_file, columns=columns)
        
    # check gt labels
    if gt_label_scope == "loc":
        # List of possible values
        gt_label_values = set(gt_label_map.keys())
        if df[gt_label].null_count() > 0:
            raise ValueError("Shouldn't be any null values in gt label col")
        unique_vals = df[gt_label].unique()
        if len(unique_vals) == 1:
            warning = f"All gt labels same for each localisation in file {input_file}"
            warnings.warn(warning)
        if not set(unique_vals).issubset(gt_label_values):
            raise ValueError("Contains gt labels outside of the domain")
        
    # if specified gt label per fov
    # 1. if there is more than one value in the column they MUST be the same!
    # 2. gt label in the scope of values
    if gt_label_scope == "fov":
        if gt_label not in set(gt_label_map.keys()):
            raise ValueError("Contains gt label outside of the domain")

    # rename
    new_names = dict(zip(columns, column_names))
    df = df.rename(new_names)

    if frame_col is None:
        df = df.with_columns(pl.lit(0).alias("frame"))
    if channel_col is None:
        df = df.with_columns(pl.lit(0).alias("channel"))

    # remove so only channels chosen remain
    if channel_col is None:
        channel_choice = [0]
    df = df.filter(pl.col("channel").is_in(channel_choice))

    # Get name of file - assumes last part of input file name
    name = os.path.basename(os.path.normpath(input_file)).removesuffix(".parquet")

    return datastruc.item(
        name,
        df,
        dim,
        channel_choice,
        channel_label,
        gt_label_scope=gt_label_scope,
        gt_label=gt_label,
        gt_label_map=gt_label_map,
    )
