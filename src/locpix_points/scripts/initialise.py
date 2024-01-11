"""Initialise project"""

import json
import socket
from importlib_resources import files
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
    data_path = os.path.relpath(data_path, start=project_directory)

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
        "data_path": data_path,  # needs to be relative to the project folder
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

    # save metadata
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(metadata_path, "w") as outfile:
        json.dump(metadata, outfile)


if __name__ == "__main__":
    main()
