### 28th June 2023

Goal: 

1. To run C15_cell data through

2. To run data downloaded from 

https://github.com/DJ-Nieves/ARI-and-IoU-cluster-analysis-evaluation/tree/main/Ground%20Truth%20-%20Scenario%2010

1. Ran:

```
python src/locpix_points/scripts/preprocess.py -i ../../../../mnt/c/Users/olive/'OneDrive - University of Leeds'/Project/output/c15_cells/annotate/annotated -c src/locpix_points/templates/preprocess.yaml -o ../../output/test -p
```
```
python src/locpix_points/scripts/process.py -i ../../output/test -c src/locpix_points/templates/process.yaml
```
```
python src/locpix_points/scripts/train.py -i ../../output/test -c src/locpix_points/templates/train.yaml
```

2. Ran

```
python src/locpix_points/scripts/preprocess.py -i tests/nieves_test_data -c src/locpix_points/templates/preprocess.yaml -o ../../output/nieves
```
```
python src/locpix_points/scripts/process.py -i ../../output/nieves -c src/locpix_points/templates/process.yaml
```
```
python src/locpix_points/scripts/train.py -i ../../output/nieves -c src/locpix_points/templates/train.yaml
```