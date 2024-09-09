# digits_letters experimental details

## Overview

This details the commands run to generate the results for the digits_letters dataset.

For each task the configuration files are located in task_ID/config and the scripts in task_ID/scripts, where ID is 1,2,...

The dataset used was 'SMLM 2D Digits 123 and TOL letters' - a dataset of either: T, O, L, 1, 2, 3 in a FOV therefore have per FOV label

Labels are:

    0: "one",
    1: "two",
    2: "three",
    3: "T",
    4: "O",
    5: "L",
    6: "grid",

Below is an overview of the details for each task 

| Task ID  | Manual features used | Model | Loc conv type | Cluster conv type |
| ------------- | ------------- | ------------- | ------------- |
| Task 1  | No  | LocClusterNet | PointNetConv | PointNetConv |
| Task 2  | No  | LocClusterNet | PointTransformer | PointTransformer |
| Task 3  | No  | LocOnlyNet | PointNet 
| Task 4  | No  | LocOnlyNet | PointTransformer 


## Note

Ran script to find that

minimum x value: -0.816
maximum x value: 0.865
minimum y value: -0.859
maximum y value: 0.748

i.e. x fov size: 1.681
     y fov size: 1.607

Obviously may be 2.0 and 2.0 but will keep these as the values for FOV size

Note the transforms have left in and removed i.e. removed x flip and y flip as otherwise is not a 3 etc. anymore

## Commands run

This assumes everything is properly installed and the user is located in the locpix-points directory.
In all commands below replace ID with the task number 1,2,3...

### Initialise

```shell
micromamba activate locpix-points
initialise
```

If first task

User name = oliver-umney
Project name = task_ID
Project saved = digits_letters
Dataset location = /mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/methods_paper/SMLM 2D Digits 123 and TOL letters/data/train
Dataset name = digits_letters
Copy preprocessed = no
.csv files = no
Already labelled = yes

If NOT first task

User name = oliver-umney
Project name = task_ID
Project saved = digits_letters
Dataset location = /mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/methods_paper/SMLM 2D Digits 123 and TOL letters/data/train
Dataset name = digits_letters
Copy preprocessed = yes
Location of the project folder = digits_letters/task_1
Copy k-fold splits = yes
Does your data already have this label = yes

### Preprocess

```shell
cd digits_letters/task_ID
bash scripts/preprocess.sh
```

### Feature extraction

```shell
bash scripts/featextract.sh
```

### Generate k-fold splits

```shell
bash scripts/generate_k_fold_splits.sh
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