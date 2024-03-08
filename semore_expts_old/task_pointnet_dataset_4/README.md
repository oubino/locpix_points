# Test locnetonly_pointnet

## Overview

THIS WAS FOR TESTING CAN DISREGARD RESULTS ETC.

Dataset used: 'Semore-Dataset 4' - Dataset of pure fibrils vs pure iso (/w 0% noise) therefore have per FOV label
               Only 1 aggregate per FOV to simplify

Note no manual features of the clusters are used in training

Use LocNetClassifyFOV

## Commands run

This assumes everything is properly installed and the user is located in the locpix-points directory.

## Note

Deleted:

fib_56 as no cluster identified
fib_71 as no cluster identified


### Initialise

```shell
micromamba activate locpix-points
initialise
```

Project name = test_pointnet
Project saved = semore_expts
Dataset name = semore_dataset_4
Dataset location = path/to/semore/data
Would you like to copy preprocessed... = no
Are your files .csv files? = no
Data should have... = yes

### Preprocess

```shell
cd semore_expts/test_pointnet
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