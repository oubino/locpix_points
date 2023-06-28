"""Annotate module

Take in items, convert to histograms, annotate,
visualise histo mask, save the exported annotation .parquet
"""

import yaml
import os
from heptapods.preprocessing import datastruc
import polars as pl
import argparse

def main():

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Annotate the data"
    )

    parser.add_argument(
        "-i", 
        "--project_direcotry", 
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
                             for annotation",
        required=True,
    )

    parser.add_argument(
        "-a",
        "--custom_annotation",
        action="store_true",
        help="if true uses custom annotation; otherwise uses manual annotation",
        required=True,
    )


    args = parser.parse_args()

    project_directory = args.project_directory

    # load yaml
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    if args.custom_annotation:
        custom_annotation(project_directory, config)
    else:
        manual_annotation(project_directory, config)

def gt_label_generator(df):

    """CUSTOM function takes in a polars dataframe and adds a
    gt_label column in whichever user specified way

        Args:
            df (polars dataframe) : Dataframe with localisations"""

    # this just takes the channel column as the ground truth label
    df = df.with_column((pl.col('channel')).alias('gt_label'))

    return df


def custom_annotation(project_directory, config):

    # list items
    try:
        files = os.listdir(os.path.join(project_directory, "preprocessed/not_annotated"))
    except FileNotFoundError:
        raise ValueError("There should be some preprocessed files to open")

    # if output directory not present create it
    output_directory = os.path.join(project_directory, "preprocessed/annotated")
    if not os.path.exists(output_directory):
        print('Making folder')
        os.makedirs(output_directory)

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(project_directory, file))

        # check no gt label already present
        if 'gt_label' in item.df.columns:
            raise ValueError('Manual segment cannot be called on a file which\
                              already has gt labels in it')

        # generate gt label
        item.df = gt_label_generator(item.df)

        # save df to parquet with mapping metadata
        # note drop zero label important is False as we have
        # channel 0 (EGFR) -> gt_label 0 -> don't want to drop this
        # drop pixel col is False as we by this point have
        # no pixel col
        item.save_to_parquet(output_directory,
                             drop_zero_label=False,
                             drop_pixel_col=False,
                             gt_label_map=config['gt_label_map'])

def manual_annotation(project_directory, config):

    # list items
    try:
        files = os.listdir(os.path.join(project_directory, "preprocessed/not_annotated"))
    except FileNotFoundError:
        raise ValueError("There should be some preprocessed files to open")

    # if output directory not present create it
    output_directory = os.path.join(project_directory, "preprocessed/annotated")
    if not os.path.exists(output_directory):
        print('Making folder')
        os.makedirs(output_directory)

    if config['dim'] == 2:
        histo_size = (config['x_bins'], config['y_bins'])
    elif config['dim'] == 3:
        histo_size = (config['x_bins'], config['y_bins'], config['z_bins'])
    else:
        raise ValueError('Dim should be 2 or 3')

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(project_directory, "preprocessed/not_annotated", file))

        # coord2histo
        item.coord_2_histo(histo_size, plot=config['plot'],
                           vis_interpolation=config['vis_interpolation'])

        # manual segment
        item.manual_segment()

        # save df to parquet with mapping metadata
        item.save_to_parquet(output_directory,
                             drop_zero_label=config['drop_zero_label'],
                             gt_label_map=config['gt_label_map'])

if __name__ == "__main__":
    def main()