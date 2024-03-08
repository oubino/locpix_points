# Task 1

## Overview

Dataset used: 'Semore' - Dataset of pure fibrils vs pure iso (/w 10% noise) therefore have per FOV label

All configuration according to task_1/config
All scripts run are located in task_1/scripts

Note no manual features of the clusters are used in training
Only the features from the PointNet are used - in the locclusternet

## Commands run

This assumes everything is properly installed and the user is located in the locpix-points directory.

### Initialise

```shell
micromamba activate locpix-points
initialise
```

Project name = task_1
Project saved = semore_expts
Dataset name = semore_dataset_1
Dataset location = path/to/semore/data

### Preprocess

```shell
cd semore_expts/task_1
bash scripts/preprocess.sh
```

### Feature extraction

```shell
bash scripts/featextract.sh
```

### K-fold 

```shell
bash scripts/k_fold.sh
```

### Feature analysis not neural-net features

```shell
bash scripts/featanalyse_manual.sh
```

### Feature analysis neural-net features

```shell
bash scripts/featanalyse_nn.sh
```