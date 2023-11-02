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

Visualisation
    - Check that loclisations are correctly connected to each other
    - visualise can start showing the features - colour coded or on the z axis maybe?

Feature analysis
    - script which analyses features of data - i.e. UMAP of user defined features
    - histograms all comparing the two classes etc.
    - this script should then also be able to take in the features we will derive from our graph neural network
    - aggregate cluster features into collated df - can then use pandas to plot this per cluster/and per patient (box plot)
    - umap of features 


ReadME
    - Visualise requires new environment to use python 3.10 - micromamba activate 
    - include how to run tests using bash tests/tests.sh 
    - also include visualise

Train
    - Write train script and update model

Misc
    - Address all warnings in code

Experiments
    - Take each cluster as a data point, have lots for cancer and lots for non cancer, if UMAP doesn't distinguish trial learning some features using PointNet or GraphNet and then use UMAP on these features - this could help - also just try linear analysis on these clusters
    - make decision on which length/area to use




-------------- 
3. Create a template notebook for each experiment (each experiment different architecture, prediction (outcome, cancernotcancer), etc.)
4. Train small test dataset 3 cancer 3 not cancer with pointtransformer
5. Implement leave one out cross validation - i.e. with remaining data train set, validation set, test set for each fold
6. GraphNet: node located at each cluster - CONSIDER HOW TO NORMALISE - MAKE SURE DOESNT JUST CONSIDER # OF POINTS + COORDINATES NEED TO BE PART OF NODE'S FEATURE I.E. X/Y MUST BE CONSIDEERED
7. GraphNet: node for each localisation only connected to locs in its cluster, message parse etc.; then alongside other features for cluster node for each cluster message parse and make final prediction - ablation study to see if improves results
8. Visualisation as a histogram + Visualisation as a pointcloud
9. Explainability stuff
10. Decision to create a reserved test set early on (how many to set aside)
11. Update ReadMe so clear how to use
12. Also add docs
13. New dataset
14. Trial making one environment combining both

1. Clustering and visualisation (feature extraction)
    a. Extra cluster features: distance birth, distance death, cluster skew?
    b. Note papers about cluster size and number of receptors per cluster for DBSCAN
    c. Also note other things can provide e.g. UMAP on features for each cluster
2. Needs to consider the different splits - script which wraps process + train + performane evaluate to do for 5-fold - create the 5 splits- pass as arugments to process which then saves as processed/fold/train, processed/fold/val, processed/fold/test after each fold clean up but keep model and splits

Do we use the geometric features just as input to embedding or should we add on

Different models to try

1. PointNet etc. on all points
2. Simple machine learning on clustered data /w features
3. GraphNet on clustered data /w features
4. GraphNet on cluster data + pointNet/graphnet? for localisations within each cluster
4.5 As above but place the top level graph randomly/systematically across the FOV?
5. Also includ learned clustering
6. UMAP of clusters and features
7. What edge features?

### 17th October 2023

Run TMA Genetec through pipeline

1. Install cuda 12.2 on WSL https://docs.nvidia.com/cuda/wsl-user-guide/index.html https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local 
2. Install requirements 

```
python src/locpix_points/scripts/preprocess.py -i ../../../../mnt/c/Users/olive/'OneDrive - University of Leeds'/Research Project/Code/tma/data/raw/locs -c src/locpix_points/templates/preprocess.yaml -o ../../output/tma_genetec -p

python src/locpix_points/scripts/featextract.py -i ../../output/tma_genetec -c src/locpix_points/templates/featextract.yaml
```


