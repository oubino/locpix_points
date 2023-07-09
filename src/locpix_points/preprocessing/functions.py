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


def file_to_datastruc(
    input_file,
    file_type,
    dim,
    channel_col,
    frame_col,
    x_col,
    y_col,
    z_col,
    channel_choice,
    channel_label=None,
    gt_label_scope=None,
    gt_label=None,
    gt_label_map=None,
    features=None,
):
    """Loads in .csv or .parquet and converts to the required datastructure.

    Currently considers the following columns: channel frame x y z
    Also user can specify the channels they want to consider, these
    should be present in the channels column

    Args:
        input_file (string) : Location of the file
        file_type (string) : Either csv or parquet
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
        gt_label (string) : If specified then this is the column with the gt_label
            in
        gt_label_map (dict) : Dictionary with keys represetning the gt label present
            in the dataset and the valu erepresenting the
            real concept e.g. 0:'dog', 1:'cat'
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

    # check file type parquet or csv
    if file_type != "csv" and file_type != "parquet":
        raise ValueError(f"{file_type} is not supported, should be csv or parquet")

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
    if gt_label is not None:
        columns.append(gt_label)
        column_names.append("gt_label")
    if len(features) != 0:
        columns.extend(features)
        column_names.extend(features)

    # Load in data
    if file_type == "csv":
        df = pl.read_csv(input_file, columns=columns)
    elif file_type == "parquet":
        df = pl.read_parquet(input_file, columns=columns)

    # List of possible values
    gt_label_values = set(gt_label_map.keys())

    # if specified gt label per loc should be:
    # 1. label for each localisation
    # 2. output warning if all gt label the same
    # 3. all gt labels in the scope of values
    if gt_label_scope == "loc":
        if df[gt_label].null_count() > 0:
            raise ValueError("Shouldn't be any null values in gt label col")
        unique_vals = df[gt_label].unique()
        if len(unique_vals) == 1:
            warning = f"All gt labels same for each localisation in file {input_file}"
            warnings.warn(warning)
        if not set(unique_vals).issubset(gt_label_values):
            raise ValueError("Contains gt labels outside of the domain")
        gt_label_fov = None

    # if specified gt label per fov
    # 1. if there is more than one value in the column they MUST be the same!
    # 2. gt label in the scope of values
    if gt_label_scope == "fov":
        unique_val = df[gt_label].unique()
        if len(unique_val) != 1:
            raise ValueError("Different labels for localisations")
        if not set(unique_val).issubset(gt_label_values):
            raise ValueError("Contains gt label outside of the domain")
        gt_label_fov = unique_val
        # drop gt label column
        df = df.drop["gt_label"]

    if gt_label_scope is None:
        gt_label_fov = None
        gt_label_map = None

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
    if file_type == "csv":
        name = os.path.basename(os.path.normpath(input_file)).removesuffix(".csv")
    elif file_type == "parquet":
        name = os.path.basename(os.path.normpath(input_file)).removesuffix(".parquet")

    return datastruc.item(
        name,
        df,
        dim,
        channel_choice,
        channel_label,
        # gt_label_scope=gt_label_scope,
        gt_label_fov=gt_label_fov,
        gt_label_map=gt_label_map,
    )
