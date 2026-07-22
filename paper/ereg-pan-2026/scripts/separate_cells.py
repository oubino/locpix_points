"""Split the annotations into respsective cells"""

import copy
import os
import numpy as np
from locpix_points.preprocessing import datastruc
import matplotlib.pyplot as plt
import polars as pl
from skimage.segmentation import watershed, flood_fill
import yaml

def extract_annotations(
        file,
        mask_img, 
        coords, 
        label_value,
        config_file,
        datastructure_folder,
        output_folder,
        cells_length,
        check=False,
        save=False,
        ):
    """Args:
        file (string): Name of the file being processed
        mask_img (numpy array): The annotations for the image
        coords (numpy array): Seeds for watershed
        label_value (float): Label value
        check (bool): Whether to check the segs
        cells_length (list): List of number of locs per cell"""

    markers = np.zeros(mask_img.shape, dtype="int32")
    # coords are (h,w) in image space
    # markers[row,col] where row is h-3:h+3 and col is w-3:w+3
    # i.e. imagine if select marker in bottom left of image
    # in image space [h,w] with origin at top this would be e.g. (470,10)
    # return coordinate (470,10) from get_coords
    # markers [467:473,7:13] is populated i.e. height of 470 ish
    # and width 10 ish is populated
    # this ensures mask_img and marker coords are in same space
    for index, coord in enumerate(coords):
        markers[coord[0] - 3 : coord[0] + 3, coord[1] - 3 : coord[1] + 3] = int(
            index + 1
        )

    # get part of mask equal to the label
    # note this converts all labels to 1
    mask_img = mask_img == label_value

    # get a mask for watershed by filling in the membrane annotations
    watershed_mask = mask_img
    for index, coord in enumerate(coords):
        coord = tuple(coord)
        watershed_mask = flood_fill(watershed_mask, coord, label_value)

    # run watershed on the mask img using the markers and only applying to relevant regions
    labels = watershed(mask_img, markers=markers, mask=watershed_mask)

    if check:

        for label in range(np.max(labels)):

            label+=1
            plt.imshow(mask_img, alpha=.8, cmap='gray')
            plt.imshow(labels==label, alpha=.8, cmap='Reds')
            plt.show()

    # load in datastructure and render into histogram
    with open(config_file, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        config = config["napari"]
    if config["dim"] == 2:
        histo_size = (config["x_bins"], config["y_bins"])
    elif config["dim"] == 3:
        histo_size = (config["x_bins"], config["y_bins"], config["z_bins"])
    else:
        raise ValueError("Dim should be 2 or 3")
    item = datastruc.item(None, None, None, None, None)
    item.load_from_parquet(os.path.join(datastructure_folder, file + ".parquet"))

    # coord2histo
    item.coord_2_histo(histo_size)

    # assign label to each localisation i.e. 1 or 2
    # first convert back to original label
    mask_img = np.where(mask_img == 1, label_value, 0)
    item.df = item.mask_pixel_2_coord(mask_img, col_name="memb_label")

    # assign cell label to each localisation
    item.df = item.mask_pixel_2_coord(labels, col_name="cell_label")

    # filter out all localisations not part of a cell
    item.df = item.df.filter(pl.col("cell_label") != 0)
    
    # partition into cells
    cells = item.df.partition_by("cell_label")

    # save each cell
    if save:
        for cell in cells:
            cell_item = copy.copy(item)
            cell_item.df = cell
            id = int(cell["cell_label"].unique().item())
            cell_item.name = cell_item.name + f"_cell_{id}"
            cell_item.save_to_parquet(
                output_folder,
                drop_zero_label=False,
                drop_pixel_col=True,
                overwrite=False
            )

            cells_length.append(len(cell))

def main():
    mask_folder = "fov/preprocessed/labels"
    label_folder = "fov/preprocessed/markers"
    datastructure_folder = "fov/preprocessed/gt_label"
    annotate_config_file = "config/annotate_cells.yaml"

    output_folder = "cells/raw"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(datastructure_folder)

    cells_length = []
    
    for file in files:

        file = file.removesuffix(".parquet")
        file = file + ".npy"        
        print("File: ", file)

        mask = np.load(os.path.join(mask_folder, file), allow_pickle=True)
        mask_img = mask.T
        
        coords = np.load(os.path.join(label_folder, file), allow_pickle=True)

        extract_annotations(
            file.removesuffix(".npy"),
            mask_img, 
            coords, 
            1,
            annotate_config_file,
            datastructure_folder,
            output_folder,
            cells_length,
            check=False,
            save=True,
        )

    print("Min/Max locs for cells", np.min(cells_length), np.max(cells_length))
    plt.hist(cells_length, bins=100)
    plt.show()
    
if __name__ == "__main__":
    main()
