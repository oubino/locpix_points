Overview
========

This repository allows for analysis of SMLM data using graph neural networks.

All code in TMA repository prepares data from ONI API (download, link with outcomes, save as .parquet wiht outcomes in metadata).

This repository then does the following:
    - Initialise a project directory
    - Feature extraction
    - Preprocess for Pytorch geometric
    - Annotate or prepare GT labels
    - Process for Pytorch geometric
    - Train
    - Evaluate
    - Allow visualisation

Installation
============

Requirements
------------

Requires Cuda 12.1 or above!

For wsl for windows - follow

Install cuda 12.2 on WSL https://docs.nvidia.com/cuda/wsl-user-guide/index.html https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

Also had to install Rapids on WSL

.. code-block:: python

    micromamba create -n rapids=23.10 -c rapidsai -c conda-forge -c nvidia cudf=23.10 cuml=23.10 python=3.10 cuda-version=12.0

Environment 1 (locpix-points)
-----------------------------

Create new environment


.. code-block:: python

    micromamba create -n locpix-points -c conda-forge python=3.11

Then install this repository

.. code-block:: python

    pip install -e .

Before installing the remaining requirements, making sure you have activated the environment first

.. code-block:: python

    pip install -r requirements.txt

Also need to install https://github.com/mims-harvard/GraphXAI

To do this clone the repository to your desired location

.. code-block:: python 

    git clone https://github.com/mims-harvard/GraphXAI.git

Then navigate to the directory and install using 

.. code-block:: python 

    pip install -e .


Environment 2 (feat_extract)
----------------------------

.. code-block:: python 

    micromamba create -n feat_extract -c rapidsai -c conda-forge -c nvidia cuml=23.10 python=3.10 cuda-version=12.0
    micromamba activate feat_extract
    pip install dask dask-ml polars pytest
    pip install -e .
    pip install torch-geometric


Note need to install locpix points as well

Environment 3 (visualise)
-------------------------

.. code-block:: python

    micromamba create -n visualise python=3.10 
    micromamba activate visualise
    pip install matplotlib numpy open3d polars torch


Quickstart (Linux)
==================

1. Initialise a project directory 

*Run*

.. code-block:: python

    initialise

2. Navigate to the project directory

3. Amend all config files

4. Preprocess the data

*Run*

.. code-block:: shell

    bash scripts/preprocess.sh

5. Extract features

*Run*

.. code-block:: shell

    bash scripts/featextract.sh

6. Run k-fold training

*Run*

.. code-block:: shell

    bash scripts/k_fold.sh


Longer description
==================

If not running on Linux or want to run a different workflow please keep reading...

The workflow above is

Workflow 1
----------

    - Initialise
    - Preprocess
    - Featextract (use Environment 2)
    - K-fold (does process + train + evaluate)

For these scripts we will need to specify the:

1. Data folder
2. Project directory
3. Location of the configuration file/folder

Recommended practice (as demosntratred in the quickstart) is to keep all configuration files for a project
in a folder inside the project directory 

project_folder/
    config/
        evaluate.yaml
        ...

Alternative workflows could look like:

Workflow 2
----------

    - Initialise
    - Preprocess
    - Featextract (use Environment 2)
    - Process
    - Train
    - Evaluate

Workflow 3
----------

    - Initialise
    - Preprocess
    - Annotate
    - Featextract (use Environment 2)
    - Process
    - Train
    - Evaluate

Feat analyse can also be run after processing and visualise can be run on the .parquet files or the processed files.
We recommend visualising the processed files as then you are able to see the graph.

Scripts
=======

The workflows consist of the scripts we need to run which are detailed below

Each script should be run with Environment 1 apart from Featextract which must be run with Environment 2 and visualise which must be run with Environment 3

Initialise
----------

*Run*

.. code-block:: python

    initialise

*Description*

Initialise a project directory, linked to the dataset you want to analyse.
Project directory contains the configuration files, scripts and metadata required.

*Structure*

Project directory/
    config/
        evaluate.yaml
        ...
    scripts/
        featextract.py
        ...
    metadata.json


Preprocess
----------

*Run*

.. code-block:: python

    preprocess


*Arguments*

    - -i Path to the input data folder
    - -c Path to configuration .yaml file
    - -o Path to the project folder will create

*Structure*

If 'gt_label_scope' in config file is null:

    - Data stored in project_folder/preprocessed/no_gt_label

If 'gt_label_scope' in config file is 'loc' or 'fov':

    - Data store in project_folder/preprocessed/gt_label

*Long description*

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

*Current limitations*

    - Currently there is no option to manually choose which channels to consider, so all channels are considered.
    - Drop zero label is set to False by default no option to change
    - Drop pixel col is set to False by default no option to change

Annotate
--------

*Run*

.. code-block:: python

    annotate


*Arguments*
    
    - -i Path to the project folder
    - -c Path to configuration .yaml file

*Structure*

Data loaded in from

    - project_folder/preprocessed/no_gt_label

Data then stored in

    - project_folder/preprocessed/gt_label

*Long description*

Each fov is visualised in a histogram, which is annotated returning localisation level labels

These are added in a separate column to the dataframe called 'gt_label'

The dataframe is saved as a .parquet file with metadata specifying the mapping from label to integer


Featextract
-----------

*Run*

.. code-block:: python

    featextract

*Arguments*

    - -i Path to the project folder
    - -c Path to configuration .yaml file

*Structure*

Data loaded in from

    - project_folder/preprocessed/gt_label

Feature data for localisations saved in

    - project_directory/preprocessed/featextract/locs

Feature data for clusters saved in

    - project_directory/preprocessed/featextract/clusters

*Long description*

For each FOV DBSCAN is used to cluster the data

Basic per-cluster features are calculated (cluster COM, localisations per cluster, radius of gyration)

PCA for each cluster is calculated (linearity, circularity)

The convex hull for each cluster is calculated (perimeter length, area, length)

The cluster density is calculated (locs/convex hull area)

*Warnings*

1. We drop all unclustered localisations
2. We drop all clusters with 2 or fewer localisations otherwise convex hull/PCA fail
3. If there are no clusters this script will fail
4. If the script drops out mid running - simply run again and it will continue from where it left off

Process
-------

*Run*

.. code-block:: python

    process

*Arguments*

    - -i Path to the project folder
    - -c Path to configuration .yaml file
    - -o (Optional) Specify output folder if not provided defaults to project_directory/processed
    - -r If you want to copy the data split of another project then include this argument with the location of the project folder
    - -m List of lists, list[0]=train files, list[1] = val files, list[2] = test files


*Structure*

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

*Long description*

A heterodataitem for each FOV is created.

This has two types of nodes: localisations and clusters.

The features for the localisations and clusters are loaded into these nodes.

Then edges are added between

    - Localisations to localisations within the same cluster
    - Localisations to the cluster they are in
    - Clusters to nearest clusters

This is then ready for training

Train
-----

*Run*

.. code-block:: python

    train


*Arguments*
    - -i Path to the project folder
    - -c Path to configuration .yaml file
    - -p (Optional) Location of processed files, if not specified defaults to project_directory/processed
    - -m (Optional) Where to store the models, if not specified defaults to project_directory/models


*Structure*

Data loaded in from

    - project_folder/processed

or

    - project_folder/{args.processed_directory}

Output model is then saved in

    - project_directory/models/

or

    - project_directory/{args.model_folder}

*Long description*

The data is loaded in, the specified model is trained and saved.


Evaluate
--------

*Run*

.. code-block:: python

    evaluate


*Arguments*
    - -i Path to the project folder
    - -c Path to configuration .yaml file
    - -m Path to the model to to evaluate
    - -p (Optional) Location of processed files, if not specified defaults to project_directory/processed
    - -e (Optional) If given then explain algorithms are run on the datas


*Structure*

Data loaded in from

    - project_folder/processed/test

or

    - project_folder/{args.processed_directory}/test

Model is loaded from 

    - {args.model_loc}


*Long description*

Data is loaded in from the test folder and the model from the model_path.
This model is then evaluated on the dataset and metrics are provided.
If the explain argument is given then explain algorithms are also run on the dataset

k-fold
------

*Run*

.. code-block:: python

    k_fold

*Arguments*

    - -i Path to the project folder
    - -c Path to folder with configuration .yaml file
    - -r (Optional) If specified this integer defines the number of random splits to perform


*Structure*

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

*Long description*

If -r flag is specified then a random split of the data occurs, otherwise the split is read from the configuration file.

For each fold, the data is processed and trained using the train and validation folds.

After each fold, the files for each FOV are removed to avoid excessive build up of files, retaining the filter_map.csv, pre_filter.pt and pre_transform.pt

Featanalyse
-----------

*Requirements*

The packages required are  installed in the locpix-points environment. These include
    - polars
    - seaborn
    - matplotlib
    - umap
    - sklearn
    - numpy

*Run*

.. code-block:: python

    featanalyse

*Arguments*

    - -i Path to the project folder
    - -c Path to configuration .yaml file
    - -n (Optional) If given then feat analysis uses the features derived by the neural net & any manual features present as well
    - -t (Optional) If present we are testing therefore use only model present in model folder, as otherwise we have to specify the model name but we won't know what it is

*Long description*

Analyse the features for the clusters, both the manual features and the ones calculated by the neural network.
This includes
  - Box plots of the features 
  - UMAP
  - Classification of the fields of view using scikit-learn
    - Logisitic regression
    - Decision trees 
    - SVM 
    - KNN  

Visualise
---------

*Run*

.. code-block:: python

    visualise

*Arguments*

    - -i Path to the file to visualise (either .parquet or .pt pytorch geometric object)
    - -x If .parquet file then name of the x column
    - -y If .parquet file then name of the y column
    - -z If .parquet and 3D then name of the z column
    - -c If .parquet name of the channel column

*Long description*

Can either load in .parquet file and visualise just the points.

Or can load in .pt pytorch geometric file and visualise the nodes and edges

Clean up
--------

Removes files ending in f".egg-info", "__pycache__", ".tox" or ".vscode"

Model architectures
===================


Pytorch geometric
=================

Currently the location is taken in as feature vector i.e. the values of x and y
Obviously may want to play with this - number of photons etc.

Pre transform: saves pretransform.pt saves the pre transform that was done to the data i.e. a knn graph of this shape

so that it can make sure the data loaded in afterwards has gone through same preprocessing


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

Need to load these in, need to calculate max and min for each column over the whole training dataset

Then can normalise features to between 0 and 1 for these features

Note that when then apply to new point need to clamp points below 0 to 0 and above 1 to 1

Then also experiment with pytorch geometric normalise features

1. Need to calculate max and min for each dataitem
2. Need to load in train/val/test files for fold 0
3. Need to normalise each feature by the max and min values
4. Then can run on arc


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
