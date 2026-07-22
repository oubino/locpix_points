Instructions to train models from scratch on our data
-----------------------------------------------------

1. Follow instructions to install and activate environment for this repository (locpix-points).

2. Navigate to the paper folder in the locpix-points directory
	1. For example: cd locpix_points/paper

3. Download data (NOTE: fov/ folder is only required if want to reproduce step 4)
	1. Download cells.zip and fov.zip folders from https://doi.org/10.5281/zenodo.21339937
	2. Unzip folders and place both in the paper/ folder

4. Extract cells from FOVs
	1. python scripts/prepare_fovs.py -i fov/raw -c config/prepare_fovs.yaml -o fov/
	2. annotate -i fov/ -c config/annotate_cells.yaml -n 
		a. When you add points, ensure no layer is selected i.e. name of layer should be "Points"
		b. Add on -r flag to relabel FOVs
	3. python scripts/separate_cells.py # Extract cells from FOV
	4. python scripts/link_cells.py  # This gives cells their GT annotation
	5. python scripts/check_cells.py  # [Optional]: Check cells have sufficient localisations

5. Visualise raw cells and invidiual raw cell
	1. jupyter-notebook  # then open the scripts/visualise.ipynb notebook, and can visualise all cells and individual raw cells

6. Neural network classification from scratch
	1. initialise -u [user-name from wandb*] -pn output -pp . -d cells/gt_label -dn ereg_cells -cp no -cs no -gt yes
		* See requirements in installation instructions
	2. for file in preprocess featextract process k_fold train evaluate; do cp -f config/"$file".yaml output/config/"${file}".yaml;done
		a. This is for linux/unix if on other may have to manually copy across preprocess, featextract, process, k_fold, train and evaluate .yaml files from config/ to output/config, replacing the files that are already there
	3. cd output
	4. python scripts/preprocess.py
	5. python scripts/featextract.py
	6. python scripts/generate_k_fold_splits.py [OPTIONAL]
		a. This will overwrite the current k_fold.yaml, which contains the splits we used in the paper to generate the results!
	7. python scripts/k_fold.py

7. At this point can visualise processed cells (clustered etc.)
	1. cd ..
	2. jupyter-notebook  # then open the scripts/visualise.ipynb notebook

8. Logistic regression classification
	1. jupyter-notebook  # then open the scripts/per_cell_simple_classification.ipynb notebook
		- This should reproduce the results of per-cell classification (deterministic)

9. Combine neural network and logistic regression classification & calculate per-patient performance
	1. python scripts/rename_nn_models.py  # Renames models 
	2. jupyter-notebook  # then open the scripts/combine_and_classify_patients.ipynb notebook
	3. python scripts/per_patient_final_results.py  # Can also be run cell by cell in some IDEs (e.g. VS Code) to display dataframes

Instructions to load in our model and evaluate on data
------------------------------------------------------

1. Follow instructions to install and activate environment for this repository (locpix-points).

2. Navigate to the paper folder in the locpix-points directory
	1. For example: cd locpix_points/paper

3. Download output data
	1. Download output.zip from https://doi.org/10.5281/zenodo.21410192
	2. Unzip folder and place in the paper/ folder

4. Visualise data
	1. jupyter-notebook  # then open the visualise.ipynb notebook

5. Combine neural network and logistic regression classification & calculate per-patient performance
	1. jupyter-notebook  # then open combine_and_classify_patients.ipynb notebook

6. Generate final confusion matrices and statistical tests
	1. python scripts/per_patient_final_results.py

Directory structure
-------------------

fov/
	raw/ --> Raw unfiltered FOVs
	preprocessed/
		no_gt_label --> Unfiltered FOVs (with some metadata)
		labels/ --> Annotations produced by napari
		markers/ --> Markers produced by napari
		gt_label/ --> FOVs with annotations i.e. segmented

cells/
	raw/ --> Raw cells
	gt_label/ --> Cells with best response label

config/ --> contains config file for scripts

scripts/
	prepare_fovs.py

environments/ --> contains copies of environments used by OU during development