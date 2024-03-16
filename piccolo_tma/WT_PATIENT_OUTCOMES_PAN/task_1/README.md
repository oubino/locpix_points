# Commands run

```
initialise
```

Please input the user name: oliver-umney
Please input the project name: task_1
Please input where you would like the project to be saved: piccolo_tma/WT_PATIENT_OUTCOMES_PAN
Select data: /mnt/c/Users/olive/OneDrive - University of Leeds/Research Project/data/piccolo_tma_ereg/linked_with_pat_outcomes/WT_CANCER_OUTCOMES_PAN/data
Please input the dataset name: piccolo_wt_patient_outcomes_panitumumab
Would you like to copy preprocessed files: no
Are your files .csv files: no
Does your data already have this label: yes

```
cd piccolo_tma/WT_PATIENT_OUTCOMES_PAN
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

STOP