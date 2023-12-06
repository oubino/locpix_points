"""Cleans up the directory removing certain files"""
import argparse
import os
import shutil

# list of file endings to remove
file_endings = [".egg-info", "__pycache__", ".tox", ".vscode"]

# folders to recursively delete
del_folders = []

# get folder to cleanup
parser = argparse.ArgumentParser(description="Clean up")

parser.add_argument(
    "-i",
    "--project_directory",
    action="store",
    type=str,
    help="the location of the project directory",
    required=True,
)


args = parser.parse_args()
project_directory = args.project_directory

# get list of folders to delete
for root, d_names, f_names in os.walk(project_directory):
    for dir in d_names:
        directory = root + "\\" + dir
        for file_ending in file_endings:
            if directory.endswith(file_ending):
                del_folders.append(root + "\\" + dir)

print(del_folders)

# delete folders
input("Are you sure (press enter to continue!)")

for folder in del_folders:
    shutil.rmtree(folder)
