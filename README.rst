Overview
========

This repository allows for analysis of SMLM data using graph neural networks.

Data can be .csv or .parquet files

This repository then does the following:
    - Initialise a project directory
    - Preprocess
    - Annotate each fov or localisation (optional)
    - Feature extraction for each cluster
    - Process for Pytorch geometric
    - Train
    - Evaluate
    - Allow visualisation
    - Analysis of manual features, neural network output, explainability etc. via jupyter notebook

Installation
============

Requirements
------------

Requires Cuda 12.1 or above!

For wsl for windows - follow

Install cuda 12.2 on WSL https://docs.nvidia.com/cuda/wsl-user-guide/index.html https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

To do: release this repository as a package on PyPI

Environment 1 (locpix-points)
-----------------------------

Create new environment

.. code-block:: python

    micromamba create -n locpix-points -c conda-forge python=3.11

Then install this repository

.. code-block:: python

    git clone https://github.com/oubino/locpix_points.git
    cd locpix_points
    pip install -e .
    cd ..

Before installing the remaining requirements, making sure you have activated the environment first

We need to install our version of pytorch geometric which we do by

.. code-block:: python

    git clone https://github.com/oubino/pytorch_geometric.git
    cd pytorch_geometric
    pip install -e .
    cd ..

Install other requirements

.. code-block:: python

    cd locpix_points
    pip install -r requirements.txt
    cd ..

Also need to install DIG

To do this clone the repository to your desired location

However, we also need a custom version of this repo fixing some bugs therefore we use our fork

.. code-block:: python 

    git clone https://github.com/oubino/DIG

Then navigate to the directory and install using 

.. code-block:: python 

    cd DIG
    pip install -e .
    cd ..


Environment 2 (feat_extract)
----------------------------

Install external packages

.. code-block:: python 

    micromamba create -n feat_extract -c rapidsai -c conda-forge -c nvidia cuml=23.10 python=3.10 cuda-version=12.2
    micromamba activate feat_extract
    pip install dask dask-ml polars pytest

Then install this repository, its additional requirements and pytorch geometric as above 

.. code-block:: python

    cd locpix_points
    pip install -e .
    cd ..

.. code-block:: python

    cd pytorch_geometric
    pip install -e .
    cd ..

.. code-block:: python

    cd locpix_points
    pip install -r requirements.txt
    cd ..

Problems
--------

You may have difficulty installing the following: open3d, torch-scatter, torch-sparse, torch-cluster

To navigate this we can 

1. Remove open3d, torch-scatter, torch-sparse and torch-cluster from requirements.txt
2. For the moment no fix for open3d
3. For torch-scatter, torch-sparse and torch-cluster - where file should be modified to the relevant file - see the torch-scatter/torch-cluster/torch-sparse github page
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

Quickstart (Linux)
==================

1. Initialise a project directory 

.. code-block:: python

    initialise

2. Navigate to the project directory

3. Amend all config files

4. Preprocess the data

.. code-block:: shell

    bash scripts/preprocess.sh

5. Annotate the data (Optional)

.. code-block:: shell

    bash scripts/annotate.sh

6. Extract features

.. code-block:: shell

    bash scripts/featextract.sh

7. Generate k-fold splits

.. code-block:: shell

    bash scripts/generate_k_fold_splits.sh

8. Run k-fold training (runs process + train + evaluate)

.. code-block:: shell

    bash scripts/k_fold.sh

9. Analyse manual and neural network features

.. code-block:: shell

    scripts/analysis.ipynb

10. Analyse locs

.. code-block:: shell

    scripts/analysis_locs.ipynb

11.  Visualise a FOV [note see Longer Description for helping set the ARGS]

.. code-block:: shell

    visualise [ARGS] 

12. Ensemble evaluate - evaluate the model running multiple times and taking an average, also allows for considering only WT cells (This is custom to our analysis) - see Longer Description for helping to set the ARGS

.. code-block:: shell

     evaluate_ensemble [ARGS]

Longer description
==================

If not running on Linux or want to run an alternative workflow we can run any of the scripts detailed below.

Each script has a configuration file, recommended practice is to keep all configuration files for the project
in a folder inside the project directory (but this is not strictly necessary!) 

::
    
    Project directory
    ├── config
    │   ├── evaluate.yaml
    │   └── ...
    └── ...

Each script should be run with Environment 1 apart from Featextract which must be run with Environment 2 

Initialise
----------

.. code-block:: python

    initialise

*Arguments*

    - -d (Optional) Path to the input data folder

If specify data folder then runs in headless mode otherwise will get data using a window

Initialise a project directory, linked to the dataset you want to analyse.
Project directory contains the configuration files, scripts and metadata required.

::
    
    Project directory
    ├── config
    │   ├── evaluate.yaml
    │   └── ...
    ├── scripts
    │   ├── featextract.py
    │   └── ...
    └── metadata.json


Preprocess
----------

.. code-block:: python

    preprocess


*Arguments*

    - -i Path to the input data folder
    - -c Path to configuration .yaml file
    - -o Path to the project folder will create

*Description*

Files are read from input data folder as .parquet files, converted to datastructures and saved as .parquet files with data in the dataframe and the following metadata

    - name: Name of the file/fov    
    - dimensions: Dimensions of the localisations
    - channels: List of ints representing channels in data user wants to consider
    - channel label: label for each channel i.e. [0:'egfr',1:'ereg',2:'unk'] means channel 0 is egfr protein, channel 1 is ereg proteins and channel 2 is unknown
    - gt_label_scope: If not specified (None) there are no gt labels. If specified then is either 'loc' - gt label per localisatoin or 'fov' - gt label for field-of-view
    - gt_label: Value of the gt label for the fov or None if gt_label_scope is None or loc
    - gt_label_map:  Dictionary with keys representing the gt label present in the dataset and the values representing the real concept e.g. 0:'dog', 1:'cat'
    - bin sizes: Size of bins of the histogram if constructed e.g. (23.2, 34.5, 21.3)

The dataframe has the following columns:

    - x
    - y
    - z
    - channel
    - frame

If 'gt_label_scope' in config file is null:

    - Data stored in project_folder/preprocessed/no_gt_label

If 'gt_label_scope' in config file is 'loc' or 'fov':

    - Data store in project_folder/preprocessed/gt_label

*Current limitations*

    - Currently there is no option to manually choose which channels to consider, so all channels are considered.
    - Drop zero label is set to False by default no option to change
    - Drop pixel col is set to False by default no option to change

Annotate
--------

.. code-block:: python

    annotate

*Arguments*
    
    - -i Path to the project folder
    - -c Path to configuration .yaml file
    - -n If specified we annotate each localisation using napari
    - -s If 'fov' we label per FOV, if 'loc' we label per localisation

*Description*

If napari:
    Each fov is visualised in a histogram, which is annotated returning localisation level labels

    These are added in a separate column to the dataframe called 'gt_label'

If fov:
    We annotate per FOV 

    This is saved in parquet metadata

If loc:
    We annotate per localisation

    This is saved in the dataframe in a column called 'gt_label'

The dataframe is saved as a .parquet file with metadata specifying the mapping from label to integer

Data loaded in from

    - project_folder/preprocessed/no_gt_label

Data then stored in

    - project_folder/preprocessed/gt_label

Featextract
-----------

USING ENVIRONMENT 2

.. code-block:: python

    featextract

*Arguments*

    - -i Path to the project folder
    - -c Path to configuration .yaml file

*Description*

For each FOV DBSCAN is used to cluster the data

Basic per-cluster features are calculated (cluster COM, localisations per cluster, radius of gyration)

PCA for each cluster is calculated (linearity, circularity)

The convex hull for each cluster is calculated (perimeter length, area, length)

The cluster density is calculated (locs/convex hull area)

Data loaded in from

    - project_folder/preprocessed/gt_label

Feature data for localisations saved in

    - project_directory/preprocessed/featextract/locs

Feature data for clusters saved in

    - project_directory/preprocessed/featextract/clusters

*Warnings*

1. We drop all unclustered localisations
2. We drop all clusters with 2 or fewer localisations otherwise convex hull/PCA fail
3. If there are no clusters this script will fail
4. If the script drops out mid running - simply run again and it will continue from where it left off

Process
-------

.. code-block:: python

    process

*Arguments*

    - -i Path to the project folder
    - -c Path to configuration .yaml file
    - -o (Optional) Specify output folder if not provided defaults to project_directory/processed
    - -r If you want to copy the data split of another project then include this argument with the location of the project folder
    - -m List of lists, list[0]=train files, list[1] = val files, list[2] = test files

*Description*

A heterodataitem for each FOV is created.

This has two types of nodes: localisations and clusters.

The features for the localisations and clusters are loaded into these nodes.

Then edges are added between

    - Localisations to localisations within the same cluster
    - Localisations to the cluster they are in
    - Clusters to nearest clusters

This is then ready for training

Data loaded in from

    - project_folder/preprocessed/featextract/locs

And

    - project_folder/preprocessed/featextact/clusters

Processed files then saved in

    - project_directory/processed/train/
    - project_directory/processed/val/
    - project_directory/processed/test/

or

    - project_directory/{args.output_folder}/train/
    - project_directory/{args.output_folder}/val/
    - project_directory/{args.output_folder}/test/

Train
-----

.. code-block:: python

    train


*Arguments*
    - -i Path to the project folder
    - -c Path to configuration .yaml file
    - -p (Optional) Location of processed files, if not specified defaults to project_directory/processed
    - -m (Optional) Where to store the models, if not specified defaults to project_directory/models


*Description*

The data is loaded in, the specified model is trained and saved.

Data loaded in from

    - project_folder/processed

or

    - project_folder/{args.processed_directory}

Output model is then saved in

    - project_directory/models/

or

    - project_directory/{args.model_folder}

Evaluate
--------

.. code-block:: python

    evaluate


*Arguments*
    - -i Path to the project folder
    - -c Path to configuration .yaml file
    - -m Path to the model to to evaluate
    - -p (Optional) Location of processed files, if not specified defaults to project_directory/processed

*Description*

Data is loaded in from the test folder and the model from the model_path.
This model is then evaluated on the dataset and metrics are provided.

Data loaded in from

    - project_folder/processed/test

or

    - project_folder/{args.processed_directory}/test

Model is loaded from 

    - {args.model_loc}

Generate k-fold splits
----------------------

.. code-block:: python

    generate_k_fold_splits.py

*Arguments*

    - -i Path to the project folder
    - -c Path to folder with configuration .yaml file
    - -s Number of splits
    - -f Whether to force and override config.yaml if already present

*Description*

Generates k-fold splits for the dataset and saves in config

Needs to be run before k-fold AND analyse_manual_features, if the latter includes classic analysis (dec tree, etc.)


k-fold
------

.. code-block:: python

    k_fold

*Arguments*

    - -i Path to the project folder
    - -c Path to folder with configuration .yaml file
    - -f Fold to start from (Optional)

*Description*

The split is read from the configuration file.

For each fold, the data is processed and trained using the train and validation folds.

After each fold, the files for each FOV are removed to avoid excessive build up of files, retaining the filter_map.csv, pre_filter.pt and pre_transform.pt

Data loaded in from

    - project_folder/preprocessed/featextract/locs

And

    - project_folder/preprocessed/featextact/clusters

Temporary processed files are saved in

    - project_directory/processed/fold_{index}/train/
    - project_directory/processed/fold_{index}/val/
    - project_directory/processed/fold_{index}/test/

However, these files are removed afterwards.

The final models are saved in

    - project_folder/models/fold_{index}/


Ensemble evaluate [custom to our analysis]
------------------------------------------

.. code-block:: python

    evaluate_ensemble

*Arguments*

- -i Path to the project folder
- -m Location of the file map for mapping files to their mutation status/outcomes
- -w (Optional) If given then only run on the WT files
- -n (Optional) Name of the model in each fold - if not given then assumes only one model present in each fold folder
- -r (Optional) Number of times to run each dataitem through the model, default = 25
- -f (Optional) Whether running for final test 

*Description*

Evaluate the model on the train/val/test sets for each fold OR alternatively for train/test if final test.
Note runs each graph through the model multiple times (default=25) and takes average 
Further, there is the option to only evaluate on the WT samples.

Analysis notebooks
------------------

analysis.ipynb and analysis_locs.ipynb allow analysis of manual features, neural network features and explainability of the algorithms.


Visualise
---------

.. code-block:: python

    visualise

*Arguments*

    - -i Path to the file to visualise (either .parquet or .pt pytorch geometric object)
    - -x If .parquet file then name of the x column
    - -y If .parquet file then name of the y column
    - -z If .parquet and 3D then name of the z column
    - -c If .parquet name of the channel column

*Description*

Can load in .pt pytorch geometric file and visualise the nodes and edges [RECOMMENDED]

OR load in .parquet file and visualise just the points.


Clean up
--------

Removes files ending in f".egg-info", "__pycache__", ".tox" or ".vscode"

Final test
----------

.. code-block:: python

    final_test

Initialise a project directory, linked to the dataset you want to analyse.
Project directory contains the configuration files, scripts and metadata required.

::
    
    Project directory
    ├── config
    │   ├── evaluate.yaml
    │   └── ...
    ├── scripts
    │   ├── featextract.py
    │   └── ...
    └── metadata.json

This is different to initialise as we now ASSUME that your input data is located as

::
    
    Input data folder
    ├── train
    │   ├── file_0.parquet
    │   └── ...
    └── test
        ├── file_0.parquet
        └── ...

*Description*

If copy files from another folder will put these in a folder "preprocessed/train" i.e. assumes copying train files.
Note can't copy files from another final_test folder for example
    
*Warning*

Currently data has to have gt_labels already loaded in

AND

There is only feature analysis of manual features

Running final test
==================

1. Initialise a new project directory 

.. code-block:: python

    final_test

*Notes*
This will create a project directory, if copy already preprocessed files then will ASSUME these are train files and place these in folder preprocessed/train

2. Navigate to the project directory

3. Amend all config files

4. Preprocess the data

.. code-block:: shell

    bash scripts/preprocess.sh

*Notes*
If the train files already copied acrossed will skip this otherwise will preprocess the train files into preprocessed/train
Will preprocess the test files into preprocessed/test

5. Annotate the data (Optional)

.. code-block:: shell

    bash scripts/annotate.sh

6. Extract features

.. code-block:: shell

    bash scripts/featextract.sh

*Notes*
Will extract features from train and test folders - similarly will skip preprocessed/train if files copied across from another folder

7. Process the data

.. code-block:: shell

    bash scripts/process.sh

8. Run training 

.. code-block:: shell

    bash scripts/train.sh


*Notes*
Trains of all the training data

9. Run evaluation

.. code-block:: shell

    bash scripts/evaluate.sh

*Notes*
Evaluate on the test set

10. Analyse manual, NN features and locs

.. code-block:: shell

    scripts/analysis.ipynb
    scripts/analysis_locs.ipynb


11.  Visualise a FOV [note see Longer Description for helping set the ARGS]

.. code-block:: shell

    visualise [ARGS] 


Mixed precision training
========================

https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam

See above link for more information.
The key takeaway is that GPUs with tensor cores can do FP16 matrix multiplications
in very optimised fashion.

Pytorch standard precision is FP32, therefore converting to FP16 can speed up
the training significantly.

However, as FP16 has a higher rounding error, small gradients can 'underflow'
to zero, where underflow means that small values become zero, which leads to
these gradients vanishing.

If we scale the gradients up, then work with them in FP16 before scaling them
back down during backpropagation we can work in FP16 while avoiding underflow.

It is called mixed precision, as we maintain two copies of a weight matrix
in FP32 and FP16.
The gradient updates are calculated using FP16 but they are applied to the
FP32 matrix, thereby making the updates safer.

Some operations are safe in FP16 while some are only safe in FP32, therefore
we work with mixed precision where pytorch automatically casts the tensors
to the safest/fastest precision.

There is memory saved from using FP16 but the speed up comes from the tensor
cores which provide faster computation for FP16 matrices.


Features of ONI data
====================

X (nm): x
Y (nm: y
Z (nm): z
X precision (nm): include, normalise to 0-1
Y precision (nm): include, normalise to 0-1
X (pix): ignore
Y (pix): ignore
Z (pix): ignore
X precision (pix): ignore
Y precision (pix): ignore
Photons: normalise 0-1
Background: normalise 0-1
PSF Sigma X (pix): normalise 0-1
PSF Sigma Y (pix): normalise 0-1
Sigma X var: normalise 0-1
Sigma Y var: normalise 0-1
p-value: leave as is

Licenses
========

+-------------------------------------+----------------------------------------------------------------------+
|               Package               |                               License                                |
+=====================================+======================================================================+
|           alabaster 0.7.13          |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             anyio 3.7.0             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           app-model 0.1.4           |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|            appdirs 1.4.4            |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|          argon2-cffi 21.3.0         |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|     argon2-cffi-bindings 21.2.0     |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             arrow 1.2.3             |                              Apache 2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|           asttokens 2.2.1           |                              Apache 2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|             attrs 23.1.0            |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             Babel 2.12.1            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            backcall 0.2.0           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|        beautifulsoup4 4.12.2        |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             bleach 6.0.0            |                       Apache Software License                        |
+-------------------------------------+----------------------------------------------------------------------+
|             build 0.10.0            |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             cachey 0.2.1            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           certifi 2023.5.7          |                               MPL-2.0                                |
+-------------------------------------+----------------------------------------------------------------------+
|             cffi 1.15.1             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|       charset-normalizer 3.1.0      |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             click 8.1.3             |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|          cloudpickle 2.2.1          |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|             cmake 3.25.0            |                              Apache 2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|              comm 0.1.3             |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|           contourpy 1.1.0           |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|            cycler 0.11.0            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            dask 2023.6.1            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            debugpy 1.6.7            |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           decorator 5.1.1           |                           new BSD License                            |
+-------------------------------------+----------------------------------------------------------------------+
|           defusedxml 0.7.1          |                                 PSFL                                 |
+-------------------------------------+----------------------------------------------------------------------+
|         docker-pycreds 0.4.0        |                          Apache License 2.0                          |
+-------------------------------------+----------------------------------------------------------------------+
|        docstring-parser 0.15        |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           docutils 0.17.1           |     public domain, Python, 2-Clause BSD, GPL 3 (see COPYING.txt)     |
+-------------------------------------+----------------------------------------------------------------------+
|           executing 1.2.0           |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|        fastjsonschema 2.17.1        |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            filelock 3.9.0           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|           fonttools 4.40.0          |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|              fqdn 1.5.1             |                               MPL 2.0                                |
+-------------------------------------+----------------------------------------------------------------------+
|          freetype-py 2.4.0          |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|           fsspec 2023.6.0           |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             gitdb 4.0.10            |                             BSD License                              |
+-------------------------------------+----------------------------------------------------------------------+
|           GitPython 3.1.31          |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            HeapDict 1.0.1           |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             hsluv 5.0.3             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|               idna 3.4              |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|            imageio 2.31.1           |                             BSD-2-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|         imageio-ffmpeg 0.4.8        |                             BSD-2-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|           imagesize 1.4.1           |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|       importlib-metadata 6.7.0      |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|            in-n-out 0.1.8           |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|           ipykernel 6.23.3          |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|            ipython 8.14.0           |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|        ipython-genutils 0.2.0       |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           ipywidgets 8.0.6          |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|         isoduration 20.11.0         |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             jedi 0.18.2             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             Jinja2 3.1.2            |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|             joblib 1.3.0            |                             BSD 3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|           jsonpointer 2.4           |                         Modified BSD License                         |
+-------------------------------------+----------------------------------------------------------------------+
|          jsonschema 4.17.3          |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            jupyter 1.0.0            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|         jupyter-client 8.3.0        |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|        jupyter-console 6.6.3        |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|          jupyter-core 5.3.1         |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|         jupyter-events 0.6.3        |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|         jupyter-server 2.7.0        |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|    jupyter-server-terminals 0.4.4   |                          # Licensing terms                           |
+-------------------------------------+----------------------------------------------------------------------+
|      jupyterlab-pygments 0.2.2      |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|       jupyterlab-widgets 3.0.7      |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|           kiwisolver 1.4.4          |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|           lazy-loader 0.3           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|      lightning-utilities 0.9.0      |                              Apache-2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|              lit 15.0.7             |                    Apache-2.0 with LLVM exception                    |
+-------------------------------------+----------------------------------------------------------------------+
|             locket 1.0.0            |                             BSD-2-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|         locpix-points 0.0.0         |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|            magicgui 0.7.2           |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|         markdown-it-py 3.0.0        |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|           MarkupSafe 2.1.3          |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|           matplotlib 3.7.2          |                                 PSF                                  |
+-------------------------------------+----------------------------------------------------------------------+
|       matplotlib-inline 0.1.6       |                             BSD 3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|             mdurl 0.1.2             |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|            mistune 3.0.1            |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|             mpmath 1.2.1            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|        mypy-extensions 1.0.0        |                             MIT License                              |
+-------------------------------------+----------------------------------------------------------------------+
|            napari 0.4.18            |                             BSD 3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|         napari-console 0.0.8        |                             BSD 3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|      napari-plugin-engine 0.2.0     |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|          napari-svg 0.1.10          |                                BSD-3                                 |
+-------------------------------------+----------------------------------------------------------------------+
|           nbclassic 1.0.0           |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|            nbclient 0.8.0           |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|           nbconvert 7.6.0           |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|            nbformat 5.9.0           |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|          nest-asyncio 1.5.6         |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             networkx 3.0            |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|            notebook 6.5.4           |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|         notebook-shim 0.2.3         |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|              npe2 0.7.0             |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|             numpy 1.25.0            |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|            numpydoc 1.5.0           |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           overrides 7.3.1           |                     Apache License, Version 2.0                      |
+-------------------------------------+----------------------------------------------------------------------+
|            packaging 23.1           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             pandas 2.0.3            |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|         pandocfilters 1.5.0         |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|             parso 0.8.3             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             partd 1.4.0             |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           pathtools 0.1.2           |                             MIT License                              |
+-------------------------------------+----------------------------------------------------------------------+
|            pexpect 4.8.0            |                             ISC license                              |
+-------------------------------------+----------------------------------------------------------------------+
|          pickleshare 0.7.5          |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             Pillow 9.3.0            |                                 HPND                                 |
+-------------------------------------+----------------------------------------------------------------------+
|              Pint 0.22              |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|              pip 23.1.2             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|          platformdirs 3.8.0         |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|            polars 0.18.5            |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             pooch 1.7.0             |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|          prettytable 3.8.0          |                            BSD (3 clause)                            |
+-------------------------------------+----------------------------------------------------------------------+
|       prometheus-client 0.17.0      |                     Apache Software License 2.0                      |
+-------------------------------------+----------------------------------------------------------------------+
|        prompt-toolkit 3.0.38        |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|           protobuf 4.23.3           |                         3-Clause BSD License                         |
+-------------------------------------+----------------------------------------------------------------------+
|             psutil 5.9.5            |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|            psygnal 0.9.1            |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|           ptyprocess 0.7.0          |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|           pure-eval 0.2.2           |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            pyarrow 12.0.1           |                     Apache License, Version 2.0                      |
+-------------------------------------+----------------------------------------------------------------------+
|            pycparser 2.21           |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           pydantic 1.10.11          |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|       pyg-lib 0.2.0+pt20cu118       |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|           Pygments 2.15.1           |                             BSD-2-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|            PyOpenGL 3.1.7           |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           pyparsing 3.0.9           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|        pyproject-hooks 1.0.0        |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             PyQt5 5.15.9            |                                GPL v3                                |
+-------------------------------------+----------------------------------------------------------------------+
|           PyQt5-Qt5 5.15.2          |                               LGPL v3                                |
+-------------------------------------+----------------------------------------------------------------------+
|          PyQt5-sip 12.12.1          |                                 SIP                                  |
+-------------------------------------+----------------------------------------------------------------------+
|          pyrsistent 0.19.3          |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|        python-dateutil 2.8.2        |                             Dual License                             |
+-------------------------------------+----------------------------------------------------------------------+
|       python-json-logger 2.0.7      |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           pytomlpp 1.0.13           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             pytz 2023.3             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           PyWavelets 1.4.1          |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|              PyYAML 6.0             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             pyzmq 25.1.0            |                               LGPL+BSD                               |
+-------------------------------------+----------------------------------------------------------------------+
|           qtconsole 5.4.3           |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|              QtPy 2.3.1             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           requests 2.31.0           |                              Apache 2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|       rfc3339-validator 0.1.4       |                             MIT license                              |
+-------------------------------------+----------------------------------------------------------------------+
|       rfc3986-validator 0.1.1       |                             MIT license                              |
+-------------------------------------+----------------------------------------------------------------------+
|             rich 13.4.2             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|         scikit-image 0.21.0         |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|          scikit-learn 1.2.2         |                               new BSD                                |
+-------------------------------------+----------------------------------------------------------------------+
|             scipy 1.11.0            |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|           Send2Trash 1.8.2          |                             BSD License                              |
+-------------------------------------+----------------------------------------------------------------------+
|          sentry-sdk 1.27.0          |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|          setproctitle 1.3.2         |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|          setuptools 68.0.0          |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|              six 1.16.0             |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             smmap 5.0.0             |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            sniffio 1.3.0            |                          MIT OR Apache-2.0                           |
+-------------------------------------+----------------------------------------------------------------------+
|        snowballstemmer 2.2.0        |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|           soupsieve 2.4.1           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             Sphinx 4.5.0            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|    sphinxcontrib-applehelp 1.0.4    |                             BSD-2-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|     sphinxcontrib-devhelp 1.0.2     |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|     sphinxcontrib-htmlhelp 2.0.1    |                             BSD-2-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|      sphinxcontrib-jsmath 1.0.1     |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|      sphinxcontrib-qthelp 1.0.3     |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
| sphinxcontrib-serializinghtml 1.1.5 |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           stack-data 0.6.2          |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            superqt 0.4.1            |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|             sympy 1.11.1            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|           terminado 0.17.1          |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|         threadpoolctl 3.1.0         |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|          tifffile 2023.7.4          |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            tinycss2 1.2.1           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             toolz 0.12.0            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|          torch 2.0.1+cu118          |                                BSD-3                                 |
+-------------------------------------+----------------------------------------------------------------------+
|    torch-cluster 1.6.1+pt20cu118    |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|        torch-geometric 2.3.1        |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|    torch-scatter 2.1.1+pt20cu118    |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|    torch-sparse 0.6.17+pt20cu118    |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|  torch-spline-conv 1.2.2+pt20cu118  |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|         torch-summary 1.4.5         |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|          torchmetrics 1.0.0         |                              Apache-2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|            tornado 6.3.2            |                              Apache-2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|             tqdm 4.65.0             |                        MPLv2.0, MIT Licences                         |
+-------------------------------------+----------------------------------------------------------------------+
|           traitlets 5.9.0           |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             triton 2.0.0            |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             typer 0.9.0             |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|       typing-extensions 4.4.0       |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|            tzdata 2023.3            |                              Apache-2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|          uri-template 1.3.0         |                             MIT License                              |
+-------------------------------------+----------------------------------------------------------------------+
|            urllib3 2.0.3            |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
|             vispy 0.12.2            |                              (new) BSD                               |
+-------------------------------------+----------------------------------------------------------------------+
|             wandb 0.15.4            |                             MIT license                              |
+-------------------------------------+----------------------------------------------------------------------+
|            wcwidth 0.2.6            |                                 MIT                                  |
+-------------------------------------+----------------------------------------------------------------------+
|            webcolors 1.13           |                             BSD-3-Clause                             |
+-------------------------------------+----------------------------------------------------------------------+
|          webencodings 0.5.1         |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|        websocket-client 1.6.1       |                              Apache-2.0                              |
+-------------------------------------+----------------------------------------------------------------------+
|             wheel 0.40.0            |                             MIT License                              |
+-------------------------------------+----------------------------------------------------------------------+
|       widgetsnbextension 4.0.7      |                         BSD 3-Clause License                         |
+-------------------------------------+----------------------------------------------------------------------+
|             wrapt 1.15.0            |                                 BSD                                  |
+-------------------------------------+----------------------------------------------------------------------+
|             zipp 3.15.0             |                               UNKNOWN                                |
+-------------------------------------+----------------------------------------------------------------------+
