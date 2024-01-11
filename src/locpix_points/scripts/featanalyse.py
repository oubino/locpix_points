"""Feature analysis module

Module takes in the .parquet files and analyses features

Config file at top specifies the analyses we want to run
"""

import argparse
import json
import os
import time

from locpix_points.data_loading import datastruc
from locpix_points.models import model_choice
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import seaborn as sns
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import torch
import warnings
import yaml


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with

    Raises:
        ValueError: If no files present to open"""

    # parse arugments
    parser = argparse.ArgumentParser(description="Analyse features")

    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="location of the project directory",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for evaluating",
        required=True,
    )

    parser.add_argument(
        "-n",
        "--neuralnet",
        action="store_true",
        help="if present then the output of the neural"
        "net is analysed rather than the manual features",
    )

    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="if present then we are testing in which case"
        "we load in the only model present",
    )

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    label_map = config["label_map"]

    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        file = os.path.basename(__file__)
        if file not in metadata:
            metadata[file] = time.asctime(time.gmtime(time.time()))
        else:
            print("Overwriting metadata...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # list items
    try:
        loc_files = os.listdir(
            os.path.join(project_directory, "preprocessed/featextract/locs")
        )
    except FileNotFoundError:
        raise ValueError("There should be some loc files to open")

    try:
        cluster_files = os.listdir(
            os.path.join(project_directory, "preprocessed/featextract/clusters")
        )
    except FileNotFoundError:
        raise ValueError("There should be some cluster files to open")

    assert loc_files == cluster_files

    # ---- Analyse cluster features -------
    if not args.neuralnet:
        analyse_manual_feats(project_directory, loc_files, label_map, config, args)
    elif args.neuralnet:
        analyse_nn_feats(project_directory, label_map, config, args)
    else:
        raise ValueError("Should be neural net or manual")


def analyse_manual_feats(
    project_directory,
    loc_files,
    label_map,
    config,
    args,
):
    """Analyse the features of the clusters manually extracted

    Args:
        project_directory (str): Location of the project directory
        loc_files (list): List of the files with the protein
            localisations
        label_map (dict): Map from the label name to number
        config (dict): Configuration for this script
        args (dict): Arguments passed to this script
    """

    # aggregate cluster features into collated df
    dfs = []

    for index, file in enumerate(loc_files):
        # loc_path = os.path.join(
        #   project_directory, f"preprocessed/featextract/locs/{file}"
        # )
        cluster_path = os.path.join(
            project_directory, f"preprocessed/featextract/clusters/{file}"
        )

        cluster_df = pq.read_table(cluster_path)

        # extract metadata
        gt_label_map = json.loads(
            cluster_df.schema.metadata[b"gt_label_map"].decode("utf-8")
        )
        gt_label_map = {int(key): value for key, value in gt_label_map.items()}
        gt_label = cluster_df.schema.metadata[b"gt_label"]
        gt_label = int(gt_label)
        label = gt_label_map[gt_label]

        # convert to polars
        cluster_df = pl.from_arrow(cluster_df)
        cluster_df = cluster_df.with_columns(pl.lit(label).alias("type"))
        cluster_df = cluster_df.with_columns(pl.lit(f"{file}").alias("file_name"))
        dfs.append(cluster_df)

    # aggregate dfs into one big df
    df = pl.concat(dfs)
    df = df.to_pandas()

    # get features present in the dataframe
    not_features = ["clusterID", "x_mean", "y_mean", "type", "file_name"]
    features = [x for x in df.columns if x not in not_features]

    # print features
    print(features)
    raise ValueError("features not chosen")

    # feature vector
    data_feats = df[features].values
    raise ValueError("this should be fit to train only")
    data_feats_scaled = StandardScaler().fit_transform(data_feats)

    # label vector
    unique_vals = sorted(df.type.unique())
    labs = sorted(label_map.keys())
    assert labs == unique_vals
    data_labels = df.type.map(label_map).values

    # file names
    names = df.file_name

    # Analyses

    # 1. Plot PCA length/area calculation vs convex hull to compare
    if config["pca_vs_convex_hull"]:
        sns.lineplot(data=df, x="length_pca", y="length_convex_hull")
        plt.show()
        sns.lineplot(data=df, x="area_pca", y="area_convex_hull")
        plt.show()

    # 2. Plot boxplots of features
    if config["boxplots"]:
        plot_boxplots(features, df)

    # 3. Plot UMAP
    if config["umap"]:
        plot_umap(data_feats_scaled, df)

    # ---------------------------------------------------------------------- #
    # Prediction methods taking in the folds
    # ---------------------------------------------------------------------- #

    X, Y, train_indices_main, val_indices_main, test_indices_main = prep_for_sklearn(
        data_feats_scaled, data_labels, names, args
    )

    # 4. Logistic regression
    if "log_reg" in config.keys():
        parameters = config["log_reg"]
        log_reg(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            features,
            parameters,
            names,
            args,
        )

    # 5. Decision tree
    if "dec_tree" in config.keys():
        parameters = config["dec_tree"]
        dec_tree(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            features,
            parameters,
            names,
            args,
        )

    # 6. SVM
    if "svm" in config.keys():
        parameters = config["svm"]
        svm(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
        )

    # 8. K-NN
    if "knn" in config.keys():
        parameters = config["knn"]
        knn(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
        )


def analyse_nn_feats(project_directory, label_map, config, args):
    """Analyse the features of the clusters from neural network

    Args:
        project_directory (str): Location of the project directory
        label_map (dict): Map from the label name to number
        config (dict): Configuration for this script
        args (dict): Arguments passed to this script

    Raises:
        ValueError: If device specified is neither cpu or gpu
    """

    # ----------------------------

    dim = config["nn_feat"]["dim"]
    if config["nn_feat"]["device"] == "gpu":
        device = torch.device("cuda")
    elif config["nn_feat"]["device"] == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("Device should be cpu or gpu")

    model_type = config["nn_feat"]["model"]

    # initialise model
    model = model_choice(
        model_type,
        # this should parameterise the chosen model
        config["nn_feat"][model_type],
        dim=dim,
        device=device,
    )

    print("\n")
    print("Loading in best model")
    print("\n")
    # needs to be from same fold as below
    fold = config["nn_feat"]["fold"]
    model_name = config["nn_feat"]["model_name"]
    if not args.test:
        model_loc = os.path.join(
            project_directory, "models", f"fold_{fold}", model_name
        )
    elif args.test:
        model_dir = os.path.join(project_directory, "models", f"fold_{fold}")
        model_name = os.listdir(model_dir)[0]
        model_loc = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(model_loc))
    model.to(device)

    # need to create a homogenous dataset consisting only of clusters from the heterogeneous graph
    data_folder = os.path.join(project_directory, "processed", "featanalysis")

    input_train_folder = os.path.join(
        project_directory, "processed", f"fold_{fold}", "train"
    )
    output_train_folder = os.path.join(data_folder, "train")
    input_val_folder = os.path.join(
        project_directory, "processed", f"fold_{fold}", "val"
    )
    output_val_folder = os.path.join(data_folder, "val")
    input_test_folder = os.path.join(
        project_directory, "processed", f"fold_{fold}", "test"
    )
    output_test_folder = os.path.join(data_folder, "test")

    output_folders = [output_train_folder, output_val_folder, output_test_folder]
    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    raise ValueError("At this moment can check if clusters have input features already")
    raise ValueError(
        "If they do have features then we shouldn't be running this analysis"
    )

    train_set = datastruc.ClusterDataset(
        input_train_folder,
        output_train_folder,
        label_level=None,
        pre_filter=None,
        save_on_gpu=None,
        transform=None,
        pre_transform=None,
        fov_x=None,
        fov_y=None,
        from_hetero_loc_cluster=True,
        loc_net=model.loc_net,
        device=device,
    )

    val_set = datastruc.ClusterDataset(
        input_val_folder,
        output_val_folder,
        label_level=None,
        pre_filter=None,
        save_on_gpu=None,
        transform=None,
        pre_transform=None,
        fov_x=None,
        fov_y=None,
        from_hetero_loc_cluster=True,
        loc_net=model.loc_net,
        device=device,
    )

    test_set = datastruc.ClusterDataset(
        input_test_folder,
        output_test_folder,
        label_level=None,
        pre_filter=None,
        save_on_gpu=None,
        transform=None,
        pre_transform=None,
        fov_x=None,
        fov_y=None,
        from_hetero_loc_cluster=True,
        loc_net=model.loc_net,
        device=device,
    )

    dataset = torch.utils.data.ConcatDataset([train_set, val_set, test_set])

    # aggregate cluster features into collated df
    dfs = []

    # load in gt_label_map
    metadata_path = os.path.join(project_directory, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # add time ran this script to metadata
        gt_label_map = metadata["gt_label_map"]

    gt_label_map = {int(key): val for key, val in gt_label_map.items()}

    for _, data in enumerate(dataset):
        # gt label
        gt_label = int(data.y)
        label = gt_label_map[gt_label]

        # file name
        file_name = data.name + ".parquet"

        # convert to polars
        data = data.x.detach().cpu().numpy()
        cluster_df = pl.DataFrame(data)
        cluster_df = cluster_df.with_columns(pl.lit(label).alias("type"))
        cluster_df = cluster_df.with_columns(pl.lit(f"{file_name}").alias("file_name"))
        dfs.append(cluster_df)

    # --------------------------

    # aggregate dfs into one big df
    df = pl.concat(dfs)
    df = df.to_pandas()

    # get features present in the dataframe
    not_features = ["type", "file_name"]
    features = [x for x in df.columns if x not in not_features]

    # feature vector
    data_feats = df[features].values
    raise ValueError("this should be fit to train only")
    data_feats_scaled = StandardScaler().fit_transform(data_feats)

    # label vector
    unique_vals = sorted(df.type.unique())
    labs = sorted(label_map.keys())
    assert labs == unique_vals
    data_labels = df.type.map(label_map).values

    # file names
    names = df.file_name

    # Analyses
    # 1. Plot boxplots of features
    if config["boxplots"]:
        plot_boxplots(features, df)

    # ---------------------------------------------------------------------- #
    # Prediction methods taking in the folds
    # ---------------------------------------------------------------------- #

    X, Y, train_indices_main, val_indices_main, test_indices_main = prep_for_sklearn(
        data_feats_scaled, data_labels, names, args
    )
    # train/test indices are list of lists
    # with one list for each fold
    train_indices_main = [train_indices_main[fold]]
    test_indices_main = [test_indices_main[fold]]

    # 2. Logistic regression
    if "log_reg" in config.keys():
        parameters = config["log_reg"]
        best_model = log_reg(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            features,
            parameters,
            names,
            args,
            fold=fold,
        )

    # 3. Decision tree
    if "dec_tree" in config.keys():
        parameters = config["dec_tree"]
        best_model = dec_tree(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            features,
            parameters,
            names,
            args,
            fold=fold,
        )

    # 4. SVM
    if "svm" in config.keys():
        parameters = config["svm"]
        best_model = svm(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
            fold=fold,
        )

    # 5. K-NN
    if "knn" in config.keys():
        parameters = config["knn"]
        best_model = knn(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
            fold=fold,
        )


def class_report(predicted, Y, names, test_indices, args, fold):
    """Produce report on classification for test set, for particular fold using
    the best model

    Args:
        predicted (array): Predicted data
        Y (array): Target data
        names (list): Names associated with each cluster
        test_indices (list): Indices of the clusters that are part
            of the test set
        args (parser args): Arguments passed to the script
        fold (int): Integer representing the fold we are evaluating on
    """

    # prediction by the best model
    df_output = pl.DataFrame({"name": names, "output": predicted, "target": Y})
    # filter dataframe by only test items
    df_output = df_output[test_indices]
    # take average prediction across all the clusters for each fov
    df_output = df_output.group_by("name").mean()
    # if average prediction is above 0.5 then predict as 1 otherwise 0
    df_output = df_output.with_columns(
        pl.when(pl.col("output") < 0.5).then(0).otherwise(1).alias("output")
    )
    # load config
    config_path = os.path.join(args.project_directory, "k_fold.yaml")
    with open(config_path, "r") as ymlfile:
        k_fold_config = yaml.safe_load(ymlfile)

    # double check that test files agree
    # splits = k_fold_config["splits"]
    # test_fold = splits["test"][fold]
    # assert (sorted(test_fold) == sorted(df_output['name'].to_list()))

    # calculate classification report
    print(f"--- Classification report (test set) for fold {fold} ---")
    y_true = df_output["target"].to_list()
    y_pred = df_output["output"].to_list()
    print(classification_report(y_true, y_pred))


def plot_boxplots(features, df):
    """Plot boxplots of the features

    Args:
        features (list): List of the features in the data to plot
        df (pl.DataFrame): Dataframe with the localisation data"""
    for f in features:
        sns.boxplot(data=df, x=f, y="type")
        plt.show()


def plot_umap(data_feats_scaled, df):
    """Plot UMAP for the features

    Args:
        data_feats_scaled (array): Features scaled between 0 and 1
        df (pl.DataFrame): Dataframe with the localisation data
    """

    warnings.warn("Not generic code")
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data_feats_scaled)

    # Plot UMAP - normal vs cancer
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[
            sns.color_palette()[x] for x in df.type.map({"normal": 0, "cancer": 1})
        ],  # edit
        label=[x for x in df.type.map({"normal": 0, "cancer": 1})],  # edit
    )
    normal_patch = mpatches.Patch(color=sns.color_palette()[0], label="Normal")  # edit
    cancer_patch = mpatches.Patch(color=sns.color_palette()[1], label="Cancer")  # edit
    plt.legend(handles=[normal_patch, cancer_patch])  # edit
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection of the dataset", fontsize=24)
    plt.show()

    # edit
    # --------------------
    # Plot UMAP patients
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[
            sns.color_palette()[x]
            for x in df.file_name.map(
                {
                    "cancer_0.parquet": 0,
                    "cancer_1.parquet": 1,
                    "cancer_2.parquet": 2,
                    "normal_0.parquet": 3,
                    "normal_1.parquet": 4,
                    "normal_2.parquet": 5,
                }
            )
        ],
        label=[
            x
            for x in df.type.map(
                {
                    "cancer_0.parquet": 0,
                    "cancer_1.parquet": 1,
                    "cancer_2.parquet": 2,
                    "normal_0.parquet": 3,
                    "normal_1.parquet": 4,
                    "normal_2.parquet": 5,
                }
            )
        ],
    )
    # lgened
    cancer_patch_0 = mpatches.Patch(color=sns.color_palette()[0], label="Cancer 0")
    cancer_patch_1 = mpatches.Patch(color=sns.color_palette()[1], label="Cancer 1")
    cancer_patch_2 = mpatches.Patch(color=sns.color_palette()[2], label="Cancer 2")
    normal_patch_0 = mpatches.Patch(color=sns.color_palette()[3], label="Normal 0")
    normal_patch_1 = mpatches.Patch(color=sns.color_palette()[4], label="Normal 1")
    normal_patch_2 = mpatches.Patch(color=sns.color_palette()[5], label="Normal 2")
    plt.legend(
        handles=[
            cancer_patch_0,
            cancer_patch_1,
            cancer_patch_2,
            normal_patch_0,
            normal_patch_1,
            normal_patch_2,
        ]
    )
    # ------------------------
    # edit
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection of the dataset", fontsize=24)
    plt.show()


def prep_for_sklearn(data_feats_scaled, data_labels, names, args):
    """Get data ready for sklearn analysis

    Args:
        data_feats_scaled (array): Features scaled between 0 and 1
        data_labels (array): Label for each data item
        names (array): Names for each data item
        args (dict): Arguments passed to the script

    Raises:
        ValueError: If overlap between train and test indices

    Returns:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of indices of train data
        val_indices_main (list): List of indices of validation data
        test_indices_main (list): List of indices of test data
    """

    warnings.warn(
        "There must be a config file for k_fold.yaml in the directory for this to work"
    )

    # load config
    config_path = os.path.join(args.project_directory, "k_fold.yaml")
    with open(config_path, "r") as ymlfile:
        k_fold_config = yaml.safe_load(ymlfile)

    splits = k_fold_config["splits"]
    train_folds = splits["train"]
    val_folds = splits["val"]
    test_folds = splits["test"]

    df_scaled = pl.DataFrame(
        {
            "X": data_feats_scaled,
            "Y": data_labels,
            "name": names,
        }
    )

    # get indices of train/test for CV
    train_indices_main = []
    val_indices_main = []
    test_indices_main = []

    for index, train_fold in enumerate(train_folds):
        val_fold = val_folds[index]
        test_fold = test_folds[index]

        train_bool = df_scaled["name"].is_in(train_fold).to_list()
        val_bool = df_scaled["name"].is_in(val_fold).to_list()
        test_bool = df_scaled["name"].is_in(test_fold).to_list()

        train_indices = np.where(train_bool)[0]
        val_indices = np.where(val_bool)[0]
        test_indices = np.where(test_bool)[0]

        train_indices_main.append(train_indices)
        val_indices_main.append(val_indices)
        test_indices_main.append(test_indices)

        if any(i in train_indices for i in test_indices):
            raise ValueError("Should not share common values")
        if any(i in train_indices for i in val_indices):
            raise ValueError("Should not share common values")
        if any(i in test_indices for i in val_indices):
            raise ValueError("Should not share common values")

    num_features = len(df_scaled["X"][0])
    print("Num features: ", num_features)
    warnings.warn(
        "Be careful, if analysing neural net features"
        "Is this the number of features you expect"
        "Did this task use manual features as well"
    )

    X = df_scaled["X"].to_list()
    Y = df_scaled["Y"].to_list()

    return X, Y, train_indices_main, val_indices_main, test_indices_main


def fold_results(X, Y, model, train_indices_main, test_indices_main, names, args, fold):
    """foo

    Args:
        X (array): Input data
        Y (array): Target data
        model (sklearn model): Model to be evaluated
        train_indices_main (list): List of the indices of the training data
        test_indices_main (list): List of the indices of the test data
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): denotes the fold we are evaluating or is None
    """

    print("---- Fit to the specified fold or each fold ----")
    # set up arrays
    X = np.array(X)
    Y = np.array(Y)
    cv = iter(zip(train_indices_main, test_indices_main))

    if fold is not None:
        assert type(fold) is int
        fold_index = fold
        evaluated = False
    else:
        fold_index = 0

    for train_fold, test_fold in cv:
        train_fold = np.array(train_fold)
        test_fold = np.array(test_fold)
        model = model.fit(X[train_fold], Y[train_fold])
        output = model.predict(X)
        class_report(
            output,
            Y,
            names,
            test_fold,
            args,
            fold_index,
        )
        fold_index += 1

        # if fold is specified should only enter iterator once
        if fold is not None:
            if not evaluated:
                evaluated = True
            else:
                raise ValueError("Error from designer")


def log_reg(
    X,
    Y,
    train_indices_main,
    val_indices_main,
    test_indices_main,
    features,
    parameters,
    names,
    args,
    fold=None,
):
    """Perform logistic reggression on the dataset

    Args:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of the indices of the training data
        val_indices_main (list): List of the indices of the validation data
        test_indices_main (list): List of the indices of the test data
        features (list): List of features analysing
        parameters (dict): Parameters to try logistic regression for
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): If specified denotes the fold we are evaluating
    """
    cv = iter(zip(train_indices_main, val_indices_main))

    model = LogisticRegression(max_iter=1000)
    clf = GridSearchCV(model, parameters, cv=cv)

    print("-----Log reg.-------")
    print("--------------------")

    clf.fit(X, Y)
    df = pl.DataFrame(clf.cv_results_)
    df = df.select(
        pl.col(
            [
                "param_C",
                "param_penalty",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
            ]
        )
    )
    print("------ Best parameters (ignore values) ---")
    print(df.sort("rank_test_score"))

    best_model = clf.best_estimator_
    best_feats = dict(zip(features, best_model.coef_[0].tolist()))
    print("------ Coefficients --------")
    print(sorted(best_feats.items(), key=lambda x: abs(x[1]), reverse=True))

    model = LogisticRegression(max_iter=1000, **clf.best_params_)
    for index, value in enumerate(train_indices_main):
        train_indices_main[index] = np.append(value, val_indices_main[index])
        if any(i in train_indices_main[index] for i in test_indices_main[index]):
            raise ValueError("Should not share common values")
    fold_results(
        X, Y, model, train_indices_main, test_indices_main, names, args, fold=fold
    )

    return best_model


def dec_tree(
    X,
    Y,
    train_indices_main,
    val_indices_main,
    test_indices_main,
    features,
    parameters,
    names,
    args,
    fold=None,
):
    """Perform decision tree on the dataset

    Args:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of the indices of the training data
        val_indices_main (list): List of the indices of the validation data
        test_indices_main (list): List of the indices of the test data
        features (list): List of features analysing
        parameters (dict): Parameters to try decision tree for
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): If specified denotes the fold we are evaluating
    """
    cv = iter(zip(train_indices_main, val_indices_main))

    model = DecisionTreeClassifier()

    clf = GridSearchCV(model, parameters, cv=cv)

    print("-----Dec tree.------")
    print("--------------------")

    clf.fit(X, Y)
    df = pl.DataFrame(clf.cv_results_)
    print(df.columns)
    df = df.select(
        pl.col(
            [
                "param_max_depth",
                "param_max_features",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
            ]
        )
    )
    print("------ Best parameters (ignore values) ---")
    print(df.sort("rank_test_score"))

    best_model = clf.best_estimator_
    print("length", best_model.feature_importances_.tolist())
    best_feats = dict(zip(features, best_model.feature_importances_.tolist()))
    print("------ Coefficients --------")
    print(sorted(best_feats.items(), key=lambda x: abs(x[1]), reverse=True))

    model = DecisionTreeClassifier(**clf.best_params_)
    for index, value in enumerate(train_indices_main):
        train_indices_main[index] = np.append(value, val_indices_main[index])
        if any(i in train_indices_main[index] for i in test_indices_main[index]):
            raise ValueError("Should not share common values")
    fold_results(
        X, Y, model, train_indices_main, test_indices_main, names, args, fold=fold
    )

    return best_model


def svm(
    X,
    Y,
    train_indices_main,
    val_indices_main,
    test_indices_main,
    parameters,
    names,
    args,
    fold=None,
):
    """Perform svm on the dataset

    Args:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of the indices of the training data
        val_indices_main (list): List of the indices of the validation data
        test_indices_main (list): List of the indices of the test data
        parameters (dict): Parameters to try svm for
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): If specified denotes the fold we are evaluating
    """

    cv = iter(zip(train_indices_main, val_indices_main))

    model = SVC()

    clf = GridSearchCV(model, parameters, cv=cv, verbose=4)

    print("--------SVM---------")
    print("--------------------")

    clf.fit(X, Y)
    df = pl.DataFrame(clf.cv_results_)
    df = df.select(
        pl.col(
            [
                "param_C",
                "param_kernel",
                "param_gamma",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
            ]
        )
    )
    print("------ Best parameters (ignore values) ---")
    print(df.sort("rank_test_score"))

    best_model = clf.best_estimator_

    model = SVC(**clf.best_params_)
    for index, value in enumerate(train_indices_main):
        train_indices_main[index] = np.append(value, val_indices_main[index])
        if any(i in train_indices_main[index] for i in test_indices_main[index]):
            raise ValueError("Should not share common values")
    fold_results(
        X, Y, model, train_indices_main, test_indices_main, names, args, fold=fold
    )

    return best_model


def knn(
    X,
    Y,
    train_indices_main,
    val_indices_main,
    test_indices_main,
    parameters,
    names,
    args,
    fold=None,
):
    """Perform knn on the dataset

    Args:
        X (array): Feature data in array
        Y (array): The labels for each data point
        train_indices_main (list): List of the indices of the training data
        val_indices_main (list): List of the indices of the validation data
        test_indices_main (list): List of the indices of the test data
        parameters (dict): Parameters to try knn for
        names (list): FOV for each cluster
        args (parser args): Args passed to script
        fold (int): If specified denotes the fold we are evaluating
    """

    cv = iter(zip(train_indices_main, val_indices_main))

    model = KNeighborsClassifier()

    clf = GridSearchCV(model, parameters, cv=cv)

    print("--------KNN---------")
    print("--------------------")

    clf.fit(X, Y)
    df = pl.DataFrame(clf.cv_results_)
    df = df.select(
        pl.col(
            [
                "param_n_neighbors",
                # "param_weights",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
            ]
        )
    )
    print("------ Best parameters (ignore values) ---")
    print(df.sort("rank_test_score"))

    best_model = clf.best_estimator_

    model = KNeighborsClassifier(**clf.best_params_)
    for index, value in enumerate(train_indices_main):
        train_indices_main[index] = np.append(value, val_indices_main[index])
        if any(i in train_indices_main[index] for i in test_indices_main[index]):
            raise ValueError("Should not share common values")
    fold_results(
        X, Y, model, train_indices_main, test_indices_main, names, args, fold=fold
    )

    return best_model


# save yaml file
# yaml_save_loc = os.path.join(project_directory, "featextract.yaml")
# with open(yaml_save_loc, "w") as outfile:
#    yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
