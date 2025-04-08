Reproducing manuscript results [~1 day]
=======================================

To reproduce results on the reserved test sets as seen in the manuscript please see below.

#. Install all pre-requisites and environment as noted on main branch

#. Switch to manuscript_version of locpix-points, by navigating to locpix-points install and switching branch

    .. code-block:: shell

        cd locpix_points
        git switch clusternet_manuscript

#. Activate the environment [replace micromamba with conda or mamba or whichever you have installed]

    .. code-block:: shell
    
        micromamba activate locpix-points

#. Download x2 .tar folder from https://doi.org/10.5281/zenodo.14246303, this includes the raw data (converted to Apache .parquet files). 

#. Extract both .tar folders

    .. code-block:: shell

        tar -zxf clusternet_hcf.tar.gz
        tar -zxf clusternet_lcf.tar.gz

#. Navigate into the folder you want to reproduce results from, e.g.

    .. code-block:: shell

            cd clusternet_hcf

#. [Optional] If you would like to re-run training or evaluation of the model, please modify the "user" in metadata.json to be your user-name from wandb.

#. [Optional] If you would like to re-run training of the model (this may slightly change results due to variability in model training), first delete or move the file in models/ folder as the models folder needs to be empty. Then run

    .. code-block:: shell
        
        python scripts/train.py

#. [Optional] If you would like to re-run evaluation of the model (this may slightly change results due to variability in sampling from the point cloud). Note there must be only one file in the models/ folder, which will be analysed.

    .. code-block:: shell
    
        python scripts/evaluate.py

#. Feature and structure analysis: launch jupyter notebook

    .. code-block:: shell
    
        jupyter-notebook

    #. [Optional] To perform feature and structure analysis, having done the optional training/evaluation of a new model, run the scripts/analysis.ipynb notebook, ensuring models/ folder has only one file, which will be analysed.
        #. To re-generate UMAP embeddings, please delete all test_umap_..._.pkl files in output/ folder.

    #. To reproduce results using the model from the manuscript. Ensure the models folder only contains the original model file that came in the download. Run the scripts/analysis_small.ipynb notebook, this allows for reproduction and visualisation of the results, including:
        #. Load in handcrafted, per-cluster and per-FOV features and visualise the UMAP representations of these. Note as UMAP is not stable (i.e. each run could produce slightly different results), the notebook loads in a previously generated UMAP plot, rather than regenerating this.
        #. Generate prediction for each item in the reserved test set and visualise the incorrect predictions in UMAP space
        #. Identify graphs closest and furthest from the centre of each class in UMAP space, and visualise the raw and clustered graphs 
        #. For these graphs visualise the results of SubgraphX on them. Note as SubgraphX is not stable (i.e. each run could produce slightly different results), the notebook loads in previously generated SubgraphX plot, rather than regenerating this.

#. [Optional] To interactively visualise Figures 2A-C and Supplementary Figure 6 interactively, download clusternet_manuscript/analysis.html and open this file in a suitable browser
