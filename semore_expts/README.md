# Semore experimental details

## Overview

This details the commands run to generate the results for the semore dataset.

For each task the configuration files are located in task_ID/config and the scripts in task_ID/scripts, where ID is 1,2,...

The dataset used was 'semore_dataset_1' - a dataset of pure fibrils vs pure iso (/w 10% noise) therefore have per FOV label

Below is an overview of the details for each task 

| Task ID  | Manual features used | Deep features used | Model |
| ------------- | ------------- | ------------- | ------------- |
| Task 1  | Yes  | No | ClusterMLP |
| Task 2  | Yes  | No | ClusterNet |
| Task 3  | No  | Yes (PointNet) | LocNet |
| Task 4  | No  | Yes (PointTransformer) | LocNet |
| Task 5  | No  | Yes (PointNet) | LocClusterNet |
| Task 6  | No  | Yes (PointTransformer) | LocClusterNet |
| Task 7  | Yes  | Yes (PointNet) | LocClusterNet |
| Task 8  | Yes  | Yes (PointTransformer) | LocClusterNet |


## Commands run

This assumes everything is properly installed and the user is located in the locpix-points directory.
In all commands below replace ID with the task number 1,2,3...

### Initialise

```shell
micromamba activate locpix-points
initialise
```

If first task

User name = YOUR-USER-NAME oliver-umney
Project name = task_ID
Project saved = semore_expts
Dataset location = PATH-TO-SEMORE-DATA  /mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/semore/data/dataset_1/train
Dataset name = semore_dataset_1
Copy preprocessed = no
.csv files = no
Already labelled = yes

If NOT first task

User name = YOUR-USER-NAME oliver-umney
Project name = task_ID
Project saved = semore_expts
Dataset location = PATH-TO-SEMORE-DATA  /mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/semore/data/dataset_1/train
Dataset name = semore_dataset_1
Copy preprocessed = yes
Location of the project folder = semore_expts/task_1
Copy k-fold splits = yes
Does your data already have this label = yes

### Preprocess

```shell
cd semore_expts/task_ID
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

### Final test

```shell
bash ERROR ERROR
```