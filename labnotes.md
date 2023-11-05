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

Important
    - Check using visualisation that scaling done correctly for cooridnates
    - For below normalise scale should be just for me as I need for all data points
    - For below ones Rotate-->Shear have option which does individual or joint option if isinstance(data, HeteroData)
    - Rewrite all the transforms for heterogeneous data  - define in this otherwise will get overwritten by future changes 
        - make sure clusters are transformed in same way so don't lose relationship between them
        - Random Rotate
        - Radom Jitter
        - Random Flip
        - Random Scale
        - Random Shear

Train
    - Write train script and update model
    
Visualisation
    - Check that loclisations are correctly connected to each other
    - visualise can start showing the features - colour coded or on the z axis maybe?

Feature analysis
    - this script should then also be able to take in the features we will derive from our graph neural network
    - tidy script up

ReadME
    - Visualise requires new environment to use python 3.10 - micromamba activate 
    - include how to run tests using bash tests/tests.sh 
    - also include visualise

Misc
    - Implement leave one out cross validation - Needs to consider the different splits - script which wraps process + train + performane evaluate to do for 5-fold - create the 5 splits- pass as arugments to process which then saves as processed/fold/train, processed/fold/val, processed/fold/test after each fold clean up but keep model and splits
    - Address all warnings in code
    - Need an entry script:
        1. Args: Name of experiment/Location for experiment/location of data for experiment
        2. Creates a folder at the location with the experiment name
        3. In this folder should be a bash script to run every step, all the template files needed, shell script for running on arc

Small dataset
    - Train pointtransformer

Experiments
    - I think we should simulate a dataset and see if the computer can derive the same features that we use (cluster diameter etc.) - use this to build/test the model a large simulated dataset - use Daniel Nieves dataset - this can also help us to validate if made any mistakes in the code
    - Take each cluster as a data point, have lots for cancer and lots for non cancer, if UMAP doesn't distinguish trial learning some features using PointNet or GraphNet and then use UMAP on these features - this could help - also just try linear analysis on these clusters
    - make decision on which length/area to use
    - GraphNet: node located at each cluster - CONSIDER HOW TO NORMALISE - MAKE SURE DOESNT JUST CONSIDER # OF POINTS + COORDINATES NEED TO BE PART OF NODE'S FEATURE I.E. X/Y MUST BE CONSIDEERED
    - GraphNet: node for each localisation only connected to locs in its cluster, message parse etc.; then alongside other features for cluster node for each cluster message parse and make final prediction - ablation study to see if improves results
    - Explainability
    - Do we use the geometric features just as input to embedding or should we add on
    - Different models to try:
        1. PointNet etc. on all points
        2. Simple machine learning on clustered data /w features
        3. GraphNet on clustered data /w features
        4. GraphNet on cluster data + pointNet/graphnet? for localisations within each cluster
        4.5 As above but place the top level graph randomly/systematically across the FOV?
        5. Also includ learned clustering
        7. What edge features?

Clustering
    - Extra cluster features: distance birth, distance death, cluster skew?
    - Note papers about cluster size and number of receptors per cluster for DBSCAN

TMA data
    - Create reserved test set early

Later
    - Trial making one environment combining both
    - Make documents


### 17th October 2023

Run TMA Genetec through pipeline

1. Install cuda 12.2 on WSL https://docs.nvidia.com/cuda/wsl-user-guide/index.html https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local 
2. Install requirements 

```
python src/locpix_points/scripts/preprocess.py -i ../../../../mnt/c/Users/olive/'OneDrive - University of Leeds'/Research Project/Code/tma/data/raw/locs -c src/locpix_points/templates/preprocess.yaml -o ../../output/tma_genetec -p

python src/locpix_points/scripts/featextract.py -i ../../output/tma_genetec -c src/locpix_points/templates/featextract.yaml
```


