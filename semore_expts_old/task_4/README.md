# Task 4

## Overview

Dataset used: 'Semore' - Dataset of pure fibrils vs pure iso (/w 10% noise) therefore have per FOV label

All configuration according to task_4/config
All scripts run are located in task_4/scripts

We use cluster features (only) and a MLP on these features for each cluster, then aggregate over all the clusters in a FOV by taking the mean, then use softmax to get the probability for the FOV of belonging to the class.

## Commands run

This assumes everything is properly installed and the user is located in the locpix-points directory.

### Initialise

```shell
micromamba activate locpix-points
initialise
```

Project name = task_4
Project saved = semore_expts
Dataset name = semore_dataset_1
Dataset location = path/to/semore/data
Copy from preprocessed = yes
	Location of the project folder = semore_expts/task_1

### Configuration  files

Modified configuration files

0. k_fold.py (removed argument that split into 5)
1. k_fold.yaml copy from task_1
2. Process
3. Train
4. Evaluate

### K-fold 

```shell
bash scripts/k_fold.sh
```