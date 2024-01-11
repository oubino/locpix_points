# Semore expts

## Task 1

Dataset 1: Dataset of pure fibrils vs pure iso (/w 10% noise) therefore have per FOV label

All configuration according to task_1/config

Note no manual features of the clusters are used in training

Only the features from the pointNet are used - in the locclusternet

Bash scripts to run are located in task_1

## Task 2

Dataset 1: Dataset of pure fibrils vs pure iso (/w 10% noise) therefore have per FOV label

We used the preprocessed/feat extracted files from task 1 and the same splits for k-fold

Now however we use manually derived cluster features + pointnet to make the classficiation

Using the locclusternet

## Task 3

Dataset 1: Dataset of pure fibrils vs pure iso (/w 10% noise) therefore have per FOV label

We used the preprocessed/feat extracted files from task 1 and the same splits for k-fold

Now however we use cluster features (only) and a cluster network to make the classficiation

## Task 3

Dataset 1: Dataset of pure fibrils vs pure iso (/w 10% noise) therefore have per FOV label

We used the preprocessed/feat extracted files from task 1 and the same splits for k-fold

Now however we use cluster features (only) and a MLP on these features for each cluster, then aggregate over all the clusters in a FOV by taking the mean, then use softmax to get the probability for the FOV of belonging to the class.

## Methods

Data generated according to semore folder - for tasks 1-3 all use same dataset 1 - which has pure iso vs pure fibrils.
These tasks

## Results


## Discussion
