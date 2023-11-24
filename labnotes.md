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

### ToDo

Semore
- When train again train not random but using splits from first one
- Explainability
- Other tasks:
    - Task 1: Dataset of pure fibrils vs pure iso (/w or /wo noise) therefore have per FOV label
    - Task 2: If successfully complete the above then, dataset of fibril + random vs fibril + iso vs random + iso
    - Task 3: Datasets with all three present but in different ratios
- Visualise to check data correctly connected

Feature analysis
    - this script should then also be able to take in the features we will derive from our graph neural network (UMAP etc.)

Testing
    - Add tests and add to GitHub actions

ReadME
    - Visualise requires new environment to use python 3.10 - micromamba activate 
    - Include visualise

Misc
    - Update evaluate so it evaluates on the test set 
    - Address all warnings in code
    - Need an entry script:
        1. Args: Name of experiment/Location for experiment/location of data for experiment
        2. Creates a folder at the location with the experiment name
        3. In this folder should be a bash script to run every step, all the template files needed, shell script for running on arc
    - make decision on which length/area to use
    - make sure doesn't just make decision based on number of points (consider how to normalise)

Other experiments
    - Varga 2023 dataset 
    - Different simulated dataset
    - Daniel Nieves dataset
    - Simple machine learning on clustered data /w features
    - Place top level graph randomly/systematically across FOV

Clustering
    - Extra cluster features: distance birth, distance death, cluster skew?
    - Note papers about cluster size and number of receptors per cluster for DBSCAN

TMA data
    - Create reserved test set early


### 17th October 2023

Run TMA Genetec through pipeline

1. Install cuda 12.2 on WSL https://docs.nvidia.com/cuda/wsl-user-guide/index.html https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local 
2. Install requirements 

```
python src/locpix_points/scripts/preprocess.py -i ../../../../mnt/c/Users/olive/'OneDrive - University of Leeds'/Research Project/Code/tma/data/raw/locs -c src/locpix_points/templates/preprocess.yaml -o ../../output/tma_genetec -p

python src/locpix_points/scripts/featextract.py -i ../../output/tma_genetec -c src/locpix_points/templates/featextract.yaml
```


