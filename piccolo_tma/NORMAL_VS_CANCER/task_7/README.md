# Commands run

```
initialise
```

Please input the user name: oliver-umney
Please input the project name: task_7
Please input where you would like the project to be saved: piccolo_tma/NORMAL_VS_CANCER
Select data: /mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/piccolo_tma_ereg/linked_with_pat_outcomes/NORMAL_VS_CANCER/data
Please input the dataset name: piccolo_normal_vs_cancer
Would you like to copy preprocessed files: yes
Would you like to copy k-fold splits from this folder: yes
Does your data already have this label: yes

```
cd piccolo_tma/NORMAL_VS_CANCER/task_7
```

Adjust preprocess.yaml

```
bash scripts/preprocess.sh
```

Adjust featextract.yaml

```
bash scripts/featextract.sh
```

Generate k-fold splits
```
bash scripts/generate_k_fold_splits.sh
```

Adjust featanalyse.yaml

```
bash scripts/featanalyse_manual.sh
```

Adjust process.yaml, train.yaml and evaluate.yaml

Run k-fold training

```
bash scripts/k_fold.sh
```
