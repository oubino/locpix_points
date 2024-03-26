# Commands run

```
initialise
```

Please input the user name: oliver-umney
Please input the project name: task_8
Please input where you would like the project to be saved: piccolo_tma/WT_PATIENT_OUTCOMES_PAN
Select data: /mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/piccolo_tma_ereg/linked_with_pat_outcomes/WT_CANCER_OUTCOMES_PAN/data
Please input the dataset name: piccolo_wt_patient_outcomes_panitumumab
Would you like to copy preprocessed files: yes
Location of the project folder: piccolo_tma/WT_PATIENT_OUTCOMES_PAN/task_1
Would you like to copy k-fold splits from this folder: yes
Does your data already have this label: yes

```
cd piccolo_tma/WT_PATIENT_OUTCOMES_PAN/task_8
```

Adjust process.yaml, train.yaml and evaluate.yaml

Run k-fold training

```
bash scripts/k_fold.sh
```
