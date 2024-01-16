"""Initialise project"""

import json
import socket
from importlib_resources import files
import polars as pl
import shutil
import time
import os
import tkinter as tk
from tkinter import filedialog


def main():
    # Get user name (needs to match weights and bias entity)
    user = input("Please input the user name (should match entity on wandbai): ")

    # Get project name/location
    project_name = input("Please input the project name: ")

    # Get project path
    project_path = input("Please input where you would like the project to be saved: ")
    project_directory = os.path.join(project_path, project_name)

    # Get dataset location
    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(title="Dataset folder")
    rel_data_path = os.path.relpath(data_path, start=project_directory)

    # Get dataset name
    dataset_name = input("Please input the dataset name: ")

    # Create project directory
    os.makedirs(project_directory)

    # Create config and scripts folder
    os.makedirs(os.path.join(project_directory, "config"))
    os.makedirs(os.path.join(project_directory, "scripts"))

    # Initialise & save metadata
    metadata = {
        "user": user,
        "project_name": project_name,
        "project_path": project_path,  # location in which project folder is created
        "data_path": rel_data_path,  # needs to be relative to the project folder
        "dataset_name": dataset_name,
        "machine": socket.gethostname(),
        "init_time": time.gmtime(time.time()),
    }

    # Copy template/config
    dir = files("locpix_points.template.config")
    dest = os.path.join(project_directory, "config")
    iterdir = dir.iterdir()
    for file in iterdir:
        shutil.copy(file, dest)

    # Copy template/scripts
    dir = files("locpix_points.template.scripts")
    dest = os.path.join(project_directory, "scripts")
    iterdir = dir.iterdir()
    for file in iterdir:
        if (
            file.name == "k_fold_initialise_split.py"
            or file.name == "k_fold_load_split.py"
            or file.name == "annotate_fov.py"
            or file.name == "annotate_loc.py"
            or file.name == "annotate_napari.py"
        ):
            continue
        shutil.copy(file, dest)

    # Copy preprocessed files from another task
    copy_preprocessed = input(
        "Would you like to copy preprocessed files from another folder - you must type yes? "
    )

    if copy_preprocessed == "yes":
        folder_loc = input("Location of the project folder: ")

        # copy preprocessed folder
        src = os.path.join(folder_loc, "preprocessed")
        dest = os.path.join(project_directory, "preprocessed")
        shutil.copytree(src, dest)

        # copy preprocess.yaml
        src = os.path.join(folder_loc, "preprocess.yaml")
        shutil.copy(src, project_directory)

        # add relevant metadata
        metadata_path = os.path.join(folder_loc, "metadata.json")
        with open(
            metadata_path,
        ) as file:
            other_metadata = json.load(file)
            # copy relevant metadata across
            metadata["preprocess.py"] = other_metadata["preprocess.py"]
            metadata["gt_label_map"] = other_metadata["gt_label_map"]

        # Copy k fold from another task
        copy_k_fold = input(
            "Would you like to copy k-fold splits from this folder - you must type yes? "
        )

        if copy_k_fold == "yes":
            # copy config
            src = os.path.join(folder_loc, "k_fold.yaml")
            dest = os.path.join(project_directory, "config/k_fold.yaml")
            shutil.copy(src, dest)

            # copy k fold script
            k_fold_src = files("locpix_points.template.scripts").joinpath(
                "k_fold_load_split.py"
            )
            dest = os.path.join(project_directory, "scripts/k_fold.py")
            shutil.copy(k_fold_src, dest)
        else:
            k_fold_src = files("locpix_points.template.scripts").joinpath(
                "k_fold_initialise_split.py"
            )
            dest = os.path.join(project_directory, "scripts/k_fold.py")
            shutil.copy(k_fold_src, dest)
    else:
        k_fold_src = files("locpix_points.template.scripts").joinpath(
            "k_fold_initialise_split.py"
        )
        dest = os.path.join(project_directory, "scripts/k_fold.py")
        shutil.copy(k_fold_src, dest)

        csvs = input(
            "Are your files .csv files - you must type yes? "
        )

        if csvs == "yes":
            
            # make data folder
            data_folder = os.path.join(project_directory, 'input_data')
            os.makedirs(data_folder)

            # load in csvs from data_path
            csv_files = os.listdir(data_path)

            for file in csv_files:
                csv_path = os.path.join(data_path, file)
                df = pl.read_csv(csv_path)
                # save as parquet files
                df.write_parquet(os.path.join(data_folder, f"{file.replace('.csv','.parquet')}"))

            # update metadata with new data path
            metadata["data_path"] = "./input_data"

    # annotate files
    annotate = input(
        "If your data does not have the label saved in the parquet file metdata already"
        "you will need to annotate the data."
        "If your data already has gt label in the parquet file enter: no"
        "or If you want to annotate each localisation using napari enter: napari"
        "or If you want to annotate each FOV not using napari enter: fov"
        "or If you want to annotate each localisation not using napari enter: loc"
    )
    if annotate == "no":
        pass
    elif annotate == "napari":
        annotate_src = files("locpix_points.template.scripts").joinpath(
            "annotate_napari.py"
        )
        dest = os.path.join(project_directory, "scripts/annotate.py")
        shutil.copy(annotate_src, dest)
    elif annotate == "fov":
        annotate_src = files("locpix_points.template.scripts").joinpath(
            "annotate_fov.py"
        )
        dest = os.path.join(project_directory, "scripts/annotate.py")
        shutil.copy(annotate_src, dest)
    elif annotate == "loc":
        annotate_src = files("locpix_points.template.scripts").joinpath(
            "annotate_loc.py"
        )
        dest = os.path.join(project_directory, "scripts/annotate.py")
        shutil.copy(annotate_src, dest)
    else:
        raise ValueError("annotate should be no, napari, fov or loc")

    # save metadata
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(metadata_path, "w") as outfile:
        json.dump(metadata, outfile)


if __name__ == "__main__":
    main()
