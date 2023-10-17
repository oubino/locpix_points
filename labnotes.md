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

### 13th October 2023

Actions

Under preprocessing

1. Clustering and visualisation (feature extraction)
    a. Use CUML to cluster calculate cluster features (skew, num locs, circularity, density, convex hull area, radius of gyration, length, distance birth, distance death) - note papers about cluster size and number of receptors per cluster for DBSCAN
    b. Also note other things can provide e.g. UMAP?
2. Add the above to the readme
3. Create a template notebook for each experiment (each experiment different architecture, prediction (outcome, cancernotcancer), etc.)
4. Train small test dataset 3 cancer 3 not cancer with pointtransformer
5. Implement leave one out cross validation - i.e. with remaining data train set, validation set, test set for each fold
6. GraphNet: node located at each cluster - CONSIDER HOW TO NORMALISE - MAKE SURE DOESNT JUST CONSIDER # OF POINTS + COORDINATES NEED TO BE PART OF NODE'S FEATURE I.E. X/Y MUST BE CONSIDEERED
7. GraphNet: node for each localisation only connected to locs in its cluster, message parse etc.; then alongside other features for cluster node for each cluster message parse and make final prediction - ablation study to see if improves results
8. Visualisation as a histogram + Visualisation as a pointcloud
9. Explainability stuff
10. Decision to create a reserved test set early on (how many to set aside)
11. Update ReadMe so clear how to use
12. New dataset

Different models to try

1. PointNet etc. on all points
2. Simple machine learning on clustered data /w features
3. GraphNet on clustered data /w features
4. GraphNet on cluster data + graph net for localisations within each cluster
4.5 As above but place the top level graph randomly/systematically across the FOV?
5. Also includ learned clustering
6. UMAP of clusters and features
7. 

### 17th October 2023

Run TMA Genetec through pipeline

```
python src/locpix_points/scripts/preprocess.py -i ../../../../mnt/c/Users/olive/'OneDrive - University of Leeds'/Research Project/Code/tma/data/raw/locs -c src/locpix_points/templates/preprocess.yaml -o ../../output/tma_genetec -p

python src/locpix_points/scripts/featextract.py -i ../../output/tma_genetec -c src/locpix_points/templates/featextract.yaml
```


