# Task 6

## Overview

Dataset used: 'Semore' - Dataset of pure fibrils vs pure iso (/w 10% noise) therefore have per FOV label

All configuration according to task_5/config
All scripts run are located in task_5/scripts

Note no manual features of the clusters are used in training
Try and use only a PointTransformer to predict per FOV 

## Commands run

This assumes everything is properly installed and the user is located in the locpix-points directory.

### Initialise

```shell
micromamba activate locpix-points
initialise
```

Project name = task_6
Project saved = semore_expts
Dataset name = semore_dataset_1
Dataset location = path/to/semore/data
Copy from preprocessed = yes
	Location of the project folder = semore_expts/task_1
Copy k-fold splits = yes

### Configuration  files

Modified configuration files

1. Process
2. Train
3. Evaluate

### K-fold 

```shell
bash scripts/k_fold.sh
```