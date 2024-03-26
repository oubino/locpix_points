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


def get_valid_response(prompt, allowed):
    while True:
        response = input(prompt)
        if response in allowed:
            break
        else:
            continue

    return response


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
        if (
            file.name == "preprocess_null.yaml"
            or file.name == "preprocess_present.yaml"
            or file.name == "annotate.yaml"
        ):
            continue
        shutil.copy(file, dest)

    # Copy template/scripts
    dir = files("locpix_points.template.scripts")
    dest = os.path.join(project_directory, "scripts")
    iterdir = dir.iterdir()
    for file in iterdir:
        if (
            file.name == "annotate_fov.py"
            or file.name == "annotate_loc.py"
            or file.name == "annotate_napari.py"
            or file.name == "annotate.sh"
        ):
            continue
        shutil.copy(file, dest)

    # Copy preprocessed files from another task
    prompt = (
        "--------------------------------------------------------------\n"
        "Would you like to copy preprocessed files from another folder?\n"
        "(yes/no): "
    )
    copy_preprocessed = get_valid_response(prompt, ["yes", "no"])

    if copy_preprocessed == "yes":
        folder_loc = input("Location of the project folder: ")

        # copy preprocessed folder
        src = os.path.join(folder_loc, "preprocessed")
        dest = os.path.join(project_directory, "preprocessed")
        shutil.copytree(src, dest)

        # copy preprocess.yaml
        src = os.path.join(folder_loc, "config/preprocess.yaml")
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
        prompt = (
            "------------------------------------------------------\n"
            "Would you like to copy k-fold splits from this folder?\n"
            "(yes/no): "
        )
        copy_k_fold = get_valid_response(prompt, ["yes", "no"])

        if copy_k_fold == "yes":
            # copy config
            src = os.path.join(folder_loc, "config/k_fold.yaml")
            dest = os.path.join(project_directory, "config/k_fold.yaml")
            shutil.copy(src, dest)
    else:
        prompt = (
            "---------------------------\n" "Are your files .csv files?\n" "(yes/no): "
        )
        csvs = get_valid_response(prompt, ["yes", "no"])

        if csvs == "yes":
            # make data folder
            data_folder = os.path.join(project_directory, "input_data")
            os.makedirs(data_folder)

            # load in csvs from data_path
            csv_files = os.listdir(data_path)

            for file in csv_files:
                csv_path = os.path.join(data_path, file)
                df = pl.read_csv(csv_path)
                # save as parquet files
                df.write_parquet(
                    os.path.join(data_folder, f"{file.replace('.csv','.parquet')}")
                )

            # update metadata with new data path
            metadata["data_path"] = "./input_data"

    # gt label for files
    prompt = (
        "-----------------------------------------------------------------\n"
        "Data should have per FOV label located in the parquet metadata OR\n"
        "Data should have per localisation label located in a column in the dataframe\n"
        "Does your data already have this label?\n"
        "(yes/no): "
    )
    gt_label_present = get_valid_response(prompt, ["yes", "no"])

    if gt_label_present == "yes":
        print("-----------------------------------\n")
        print(
            "Preprocess .yaml needs to be adjusted if you haven't copied preprocessed files!"
        )
        # don't need to copy annotate but copy correct preprocess
        src = files("locpix_points.template.config").joinpath("preprocess_present.yaml")
        dest = os.path.join(project_directory, "config/preprocess.yaml")
        shutil.copy(src, dest)
    else:
        # copy preprocess file
        src = files("locpix_points.template.config").joinpath("preprocess_null.yaml")
        dest = os.path.join(project_directory, "config/preprocess.yaml")
        shutil.copy(src, dest)

        # copy annotate bash script and yaml
        src = files("locpix_points.template.scripts").joinpath("annotate.sh")
        dest = os.path.join(project_directory, "scripts/annotate.sh")
        shutil.copy(src, dest)
        src = files("locpix_points.template.config").joinpath("annotate.yaml")
        dest = os.path.join(project_directory, "config/annotate.yaml")
        shutil.copy(src, dest)

        prompt = (
            "-----------------------------------\n"
            "You will need to annotate the data.\n"
            "annotate.yaml needs to be adjusted.\n"
            "If you want to annotate each localisation enter: loc\n"
            "OR If you want to annotate each FOV enter: fov\n"
            "(fov/loc): "
        )
        annotate = get_valid_response(prompt, ["fov", "loc"])

        if annotate == "fov":
            # copy annotate file
            annotate_src = files("locpix_points.template.scripts").joinpath(
                "annotate_fov.py"
            )
            dest = os.path.join(project_directory, "scripts/annotate.py")
            shutil.copy(annotate_src, dest)
        elif annotate == "loc":
            prompt = (
                "-----------------------------------\n"
                "Do you want to annotate using napari?\n"
                "(yes/no): "
            )
            napari = get_valid_response(prompt, ["yes", "no"])

            if napari == "yes":
                # copy annotate file
                annotate_src = files("locpix_points.template.scripts").joinpath(
                    "annotate_napari.py"
                )
                dest = os.path.join(project_directory, "scripts/annotate.py")
                shutil.copy(annotate_src, dest)
            else:
                # copy annotate file
                annotate_src = files("locpix_points.template.scripts").joinpath(
                    "annotate_loc.py"
                )
                dest = os.path.join(project_directory, "scripts/annotate.py")
                shutil.copy(annotate_src, dest)

    # save metadata
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(metadata_path, "w") as outfile:
        json.dump(metadata, outfile)


if __name__ == "__main__":
    main()
