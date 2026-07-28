Instructions to train models from scratch on our data or re-generate preprocessing results
------------------------------------------------------------------------------------------

1. Follow instructions to install and activate environment for this repository (locpix-points).

2. Navigate to the paper folder in the locpix-points directory
	2.1. For example: cd locpix_points/paper/ereg-pan-2026

3. Download data (NOTE: fov/ folder is only required if want to reproduce step 4)
	3.1. Download cells.zip and fov.zip folders from https://doi.org/10.5281/zenodo.21397223
	3.2. Unzip folders and place both in the paper/ereg-pan-2026/ folder

4. Extract cells from FOVs
	4.1 python scripts/prepare_fovs.py -i fov/raw -c config/prepare_fovs.yaml -o fov/
	4.2A. To visualise existing annotations and reannotate if desired:
		a. annotate -i fov/ -c config/annotate_cells.yaml -n -r
	4.2B. To annotate from scratch without seeing previous annotations:
		a. From fov/preprocessed/ remove gt_label/, labels/, and markers/
		b. annotate -i fov/ -c config/annotate_cells.yaml -n
	
	4.3. If annotating:
	    a. FOV appears in napari 
		b. Pixel sizes (xy bins) for the visualisation (approx. 100 nm) appear in the terminal
		c. Delete previous annotation layers if relevant
			- If only visualising previous annotations, not reannotating, leave them as they are (ignore e, f)
		d. Adjusting gamma may help with visualisation
		e. In napari, use a paintbrush in a "Labels" layer to annotate membranes  # Stored in fov/preprocessed/labels
		f. In a "Points" layers, add a point in the interior of each annotated cell  # Stored in fov/preprocessed/markers
		g. Close napari; the next cell will appear ready for annotation; repeat a-e

	4.4A. To use previously annotated and extracted cell data:
		- Continue to step 5.
	4.4B. To re-extract existing cells or extract newly annotated cells from fovs, rather than using downloaded cell data:
		a. Remove all of the cells/ folder
		b. python scripts/separate_cells.py  # Extract cells from FOV
			- Annotates whole cells based on filling with watershed from the interior points to the membranes
			- Displays a histogram of localisation count per annotated cell
			- Close the histogram window to continue
		c. python scripts/link_cells.py  # This gives cells their clinical GT annotation
		d. python scripts/check_cells.py  # [Optional]: Check cells have sufficient localisations

5. Visualise raw cells and individual raw cells
	5.1. jupyter-notebook  # then open the scripts/visualise.ipynb notebook, and can visualise all cells and individual raw cells
	- Visualising procesed data will not work until after step 6.

6. Neural network classification from scratch
	6.1. initialise -u [user-name from wandb*] -pn output -pp . -d cells/gt_label -dn ereg_cells -cp no -cs no -gt yes
		* See requirements in installation instructions
		- The next step addresses the message about preprocess.yaml and adjusting configuration files.
	6.2. for file in preprocess featextract process k_fold train evaluate; do cp -f config/"$file".yaml output/config/"${file}".yaml; done
		a. This is for linux/unix if on other may have to manually copy across preprocess, featextract, process, k_fold, train and evaluate .yaml files from config/ to output/config, replacing the files that are already there
	6.3. cd output
	6.4. python scripts/preprocess.py
	6.5. python scripts/featextract.py
	6.6. [OPTIONAL] Generate new k-fold validation splits: This will replace the current k_fold.yaml, which contains the splits we used in the paper to generate the results!
		a. If output/config/k_fold.yaml has been generated previously, remove it
		b. python scripts/generate_k_fold_splits.py
	6.7. python scripts/k_fold.py  # Train models (Ignore the message from pytorch-geomtric about pre-filtering)

7. At this point we can visualise processed cells (clustered etc.)
	7.1. cd ..
	7.2. jupyter-notebook  # then open the scripts/visualise.ipynb notebook

8. Logistic regression classification
	8.1. jupyter-notebook  # then open the scripts/per_cell_simple_classification.ipynb notebook
		- This should reproduce the results of per-cell classification (deterministic)

9. Combine neural network and logistic regression classification & calculate per-patient performance
	9.1. python scripts/rename_nn_models.py  # Renames models 
	9.2. jupyter-notebook  # then open the scripts/combine_and_classify_patients.ipynb notebook and run to the end
	9.3. python scripts/per_patient_final_results.py  # Can also be run cell by cell in some IDEs (e.g. VS Code) to display dataframes

Instructions to load in our model and evaluate on data
------------------------------------------------------

1. Follow instructions to install and activate environment for this repository (locpix-points).

2. Navigate to the paper folder in the locpix-points directory
	2.1. For example: cd locpix_points/paper/ereg-pan-2026

3. Download output data
	3.1. Download output.zip from https://doi.org/10.5281/zenodo.21410192
	3.2. Unzip folder and place in the paper/ereg-pan-2026/ folder

4. Visualise data
	4.1. jupyter-notebook  # then open the scripts/visualise.ipynb notebook

5. Combine neural network and logistic regression classification & calculate per-patient performance
	5.1. jupyter-notebook  # then open the scripts/combine_and_classify_patients.ipynb notebook and run to the end

6. Generate final confusion matrices and statistical tests
	6.1. python scripts/per_patient_final_results.py

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

environments/ --> contains copies of environments used during development