Overview
========

This repository allows for classification of SMLM data (.csv or .parquet files) using graph neural networks and analysis of the features and structures that led to the classification.

The main steps in the pipeline are:
    - Initialise a project directory
    - Preprocess the data
    - Annotate each fov or localisation (optional)
    - Extract features for each cluster
    - Process for Pytorch geometric
    - Train a classification model
    - Evaluate the classification model
    - Visualise the raw data and graphs
    - Analysis of manual features, neural network output, explainability etc. via jupyter notebooks

Installation [~30 mins - 1 hour]
================================

Requirements
------------

* Tested on Windows computer using windows subsytem for linux (WSL) 2 with a NVIDIA GPU
* Requires a CUDA-capable GPU
* Require Cuda 12.1+
    * For WSL for windows, follow: https://docs.nvidia.com/cuda/wsl-user-guide/index.html https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
* Require `micromamba <https://mamba.readthedocs.io/en/latest/>`_ [recommended] or anaconda/miniconda/mamba
    * For all commands below replace micromamba with conda etc. depending on which you have installed
* wandb for training/evaluating models
    * Set up an account and login using instructions at `wandb <https://docs.wandb.ai/quickstart/>`_
    * Make a note of your user-name for future use
* Requires environments 1 and 2 below

Environment 1 (locpix-points)
-----------------------------

Create and activate new environment

.. code-block:: python

    micromamba create -n locpix-points -c conda-forge python=3.11
    micromamba activate locpix-points

Install this repository

.. code-block:: python

    git clone https://github.com/oubino/locpix_points.git
    cd locpix_points
    pip install -e .
    cd ..

Install our version of pytorch geometric

.. code-block:: python

    git clone https://github.com/oubino/pytorch_geometric.git
    cd pytorch_geometric
    pip install -e .
    git checkout hetero_transforms
    cd ..

Install other requirements [note may have to change the wheels to install based on which cuda you have i.e. cu121 = cuda 12.1]

.. code-block:: python

    pip install open3d torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html torch-summary torchmetrics pytest --no-cache-dir

Install DIG

.. code-block:: python 

    git clone https://github.com/oubino/DIG
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

    pip install open3d torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html torch-summary torchmetrics pytest --no-cache-dir

Problems
--------

You may have difficulty installing the following: open3d, torch-scatter, torch-sparse, torch-cluster

To navigate this we can 

1. Do not install open3d
2. For torch-scatter, torch-sparse and torch-cluster run the following (where file should be modified to the relevant file - see the torch-scatter/torch-cluster/torch-sparse github page)

.. code-block:: python

    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

Demo (On small dataset) (~1-2 hours with a GPU)
===============================================

This includes 50 items from each class from the digits and letters dataset in the folder data/ which will be used to demo the pipeline.

All pre-requisites and environments need to be installed as above.

The following commands can then be run on the command line.

#. Change directory to locpix-points/demo folder

    .. code-block:: shell

        cd locpix_points/demo

#. Initialise

    .. code-block:: shell

        micromamba activate locpix-points
        initialise
    
    * User name = [user-name from wandb]
    * Project name = output
    * Project saved = .
    * Dataset location = demo/data
    * Dataset name = demo
    * Copy preprocessed = no
    * .csv files = no
    * Already labelled = yes

    This will generate a folder called output/ where we will be analysing the data.

#. Replace output/config files with files in demo/config

    .. code-block:: shell

        cp -rf config output/


#. Preprocess

    .. code-block:: shell

        cd output
        bash scripts/preprocess.sh
    

    This preprocesses the data into a folder preprocessed/

#. Feature extraction

    .. code-block:: shell

        bash scripts/featextract.sh


    This extracts features from the data into a folder preprocessed/featextract

#. Generate k-fold splits

    .. code-block:: shell

        bash scripts/generate_k_fold_splits.sh
    

    This generates a file k_fold.yaml in config/ containing the splits

#. K-fold [remove -w flag to scripts/k_fold.py in main_k if want to run with wandb]

    .. code-block:: shell

        bash scripts/k_fold.sh
    

    This performs k-fold training, generating models in models/ folder

#. Then can analyse features using
    
    * In config/featanalyse_nn.yaml, modify model_name to be the name of the model to analyse in the models/ folder 
    e.g. If the model in the models folder is called test.pt, replace INSERTMODELNAME with test

    .. code-block:: shell

        jupyter-notebook

    * Run analysis notebook: scripts/analysis.ipynb
    * Do not run any "patient" cells

#.  [Visualise a FOV]
    
    .. code-block:: shell
    
         visualise [ARGS]

    * Generates a window visualising the file
    
    *Arguments*

        - -i Path to the file to visualise (either .parquet or .pt pytorch geometric object)
        - -x If .parquet file then name of the x column
        - -y If .parquet file then name of the y column
        - -z If .parquet and 3D then name of the z column
        - -c If .parquet name of the channel column


Reproducing manuscript results [~1 day]
=======================================

To reproduce results on the reserved test sets as seen in the manuscript please see below.

#. Install all pre-requisites and environments as above, must have micromamba installed (alternatively can amend the .sh scripts to use conda instead)

#. Switch to manuscript_version of locpix-points, by navigating to locpix-points install and switching branch

    .. code-block:: shell

        cd locpix_points
        git checkout clusternet_manuscript

#. Download x2 .tar folder from https://doi.org/10.5281/zenodo.14246303, this includes the raw data (converted to Apache .parquet files). 

#. Extract both .tar folders

    .. code-block:: shell

        tar -zxf clusternet_hcf.tar.gz
        tar -zxf clusternet_lcf.tar.gz

#. Navigate into the folder you want to reproduce results from, e.g.

    .. code-block:: shell

            cd clusternet_hcf

#. [Optional] If you would like to re-run training or evaluation of the model, please modify the "user" in metadata.json to be your user-name from wandb.

#. [Optional] If you would like to re-run training of the model (this may slightly change results due to variability in model training), first delete or move the file in models/ folder as the models folder needs to be empty
Then run

    .. code-block:: shell
        
        bash scripts/train.sh

#. [Optional] If you would like to re-run evaluation of the model (this may slightly change results due to variability in sampling from the point cloud). 
Note there must be only one file in the models/ folder, which will be analysed.

    .. code-block:: shell
    
        bash scripts/evaluate.sh

#. Feature and structure analysis: activate the correct environment and launch jupyter notebook
    .. code-block:: shell
    
        micromamba activate locpix-points
        jupyter-notebook

    #. [Optional] To perform feature and structure analysis, having done the optional training/evaluation of a new model, run the scripts/analysis.ipynb notebook, ensuring models/ folder has only one file, which will be analysed.

    #. To reproduce results using the model from the manuscript. Ensure the models folder only contains the original model file that came in the download. Run the scripts/analysis_small.ipynb notebook, this allows for reproduction and visualisation of the results, including:
        #. Load in handcrafted, per-cluster and per-FOV features and visualise the UMAP representations of these. Note as UMAP is not stable (i.e. each run could produce slightly different results), the notebook loads in a previously generated UMAP plot, rather than regenerating this.
        #. Generate prediction for each item in the reserved test set and visualise the incorrect predictions in UMAP space
        #. Identify graphs closest and furthest from the centre of each class in UMAP space, and visualise the raw and clustered graphs 
        #. For these graphs visualise the results of SubgraphX on them. Note as SubgraphX is not stable (i.e. each run could produce slightly different results), the notebook loads in previously generated SubgraphX plot, rather than regenerating this.

#. [Optional] To interactively visualise Figures 2A-C and Supplementary Figure 6 interactively, download clusternet_manuscript/analysis.html and open this file in a suitable browser

Other commands (Linux)
======================

#. After preprocessing and before feature extraction can annotate the data (Optional)

    .. code-block:: shell
    
        bash scripts/annotate.sh


#. Can analyse the localisations

    .. code-block:: shell
    
        scripts/analysis_locs.ipynb


#. Evaluate the model multiple times and take an average - ARGS see longer description in `errata <https://github.com/oubino/locpix_points/blob/main/errata.rst>`_.

    .. code-block:: shell
    
         evaluate_ensemble [ARGS]

Errata
======

For more information, including a longer description of each command see `errata <https://github.com/oubino/locpix_points/blob/main/errata.rst>`_.
