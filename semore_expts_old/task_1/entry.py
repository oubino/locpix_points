"""Initialise project"""

import json
import socket
from importlib_resources import files
import shutil
import time
import os

def main():
    
    # Get project name/location
    project_name = input('Please input the project name: ')

    # Get project path
    project_path = input('Please input where you would like the project to be saved: ')
    project_directory = os.path.join(project_path, project_name)

    # Get dataset location
    data_path = input('Please input the dataset location: ')
    data_path = os.path.relpath(data_path, start=project_directory)

    # Get dataset name
    dataset_name = input('Please input the dataset name: ')

    # Create project directory
    os.mkdir(project_directory)

    # Create config and scripts folder
    os.mkdir(project_directory, 'config')
    os.mkdir(project_directory, 'scripts')

    # Initialise & save metadata
    metadata = {
        'project_name': project_name,
        'project_path': project_path, # location in which project folder is created
        'data_path': data_path, # needs to be relative to the project folder
        'dataset_name': dataset_name,
        'machine': socket.gethostname(),
        'init_time': time.gmtime(time.time()),
    }

    # save metadata
    metadata_path = os.path.join(project_directory, 'metadata.json')
    with open(metadata_path, "w") as outfile:
        json.dump(metadata, outfile)
        
    # Copy template/config
    dir = files('locpix_points.template.config')
    dest = os.path.join(project_directory, 'config')
    iterdir = dir.iterdir()
    for file in iterdir:
        shutil.copy(file, dest)

    # Copy template/scripts
    dir = files('locpix_points.template.scripts')
    dest = os.path.join(project_directory, 'scripts')
    iterdir = dir.iterdir()
    for file in iterdir:
        shutil.copy(file, dest)
    

if __name__ == "__main__":
    main()



