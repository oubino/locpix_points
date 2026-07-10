Instructions
------------

1. COPY DIRECTORY


2. Follow instructions to install and activate environment at https://github.com/oubino/locpix_points

3. NAVIGATE TO DIRECTORY [paper]

4. Extract cells from FOVs
	1. python scripts/prepare_fovs.py -i fov/raw -c config/prepare_fovs.yaml -o fov/
	2. annotate -i fov/ -c config/annotate_cells.yaml -n 
		a. When you add points, ensure no layer is selected i.e. name of layer should be "Points"
		b. Add on -r flag to relabel FOVs
	3. python scripts/separate_cells.py # Extract cells from FOV
	4. python scripts/link_cells.py # This gives cells their GT annotation
	5. python scripts/check_cells.py # [Optional]: Check cells have sufficient localisations

5. Visualise raw cells and invidiual raw cell
	1. jupyter-notebook # then open the visualise.ipynb notebook, and can visualise all cells and individual raw cell

6. Neural network classification from scratch
	1. initialise -u oliver-umney -pn output -pp . -d cells/gt_label -dn ereg_cells -cp no -cs no -gt yes
	2. for file in preprocess featextract process k_fold train evaluate; do cp -f config/"$file".yaml output/config/"${file}".yaml;done
		a. This is for linux/unix if on other may have to manually copy across preprocess, featextract, process, k_fold, train and evaluate .yaml files from config/ to output/config, replacing the files that are already there
	3. cd output
	4. python scripts/preprocess.py
	5. python scripts/featextract.py
	6. python scripts/generate_k_fold_splits.py [OPTIONAL]
		a. This will overwrite the current k_fold.yaml, which contains the splits we used in the paper to generate the results!
	7. python scripts/k_fold.py

7. At this point can visualise processed cells (clustered etc.)
	1. jupyter-notebook # then open the visualise.ipynb notebook,

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