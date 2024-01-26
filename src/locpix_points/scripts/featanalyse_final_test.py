"""Feature analysis module for the final test

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
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import seaborn as sns
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
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
        ValueError: If no files present to open
        NotImplementedError: Temporary if try to
            do neural network"""

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
        "-a",
        "--automatic",
        action="store_true",
        help="if present then there should be only one model present in the folder"
        "which we load in",
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
        loc_train_files = os.listdir(
            os.path.join(project_directory, "preprocessed/train/featextract/locs")
        )
        loc_test_files = os.listdir(
            os.path.join(project_directory, "preprocessed/test/featextract/locs")
        )
    except FileNotFoundError:
        raise ValueError("There should be some loc files to open")

    try:
        cluster_train_files = os.listdir(
            os.path.join(project_directory, "preprocessed/train/featextract/clusters")
        )
        cluster_test_files = os.listdir(
            os.path.join(project_directory, "preprocessed/test/featextract/clusters")
        )
    except FileNotFoundError:
        raise ValueError("There should be some cluster files to open")

    assert loc_train_files == cluster_train_files
    assert loc_test_files == cluster_test_files

    # make seaborn plots pretty
    sns.set_style("darkgrid")

    # make output folder
    output_folder = os.path.join(project_directory, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- Analyse cluster features -------
    if not args.neuralnet:
        analyse_manual_feats(
            project_directory,
            cluster_train_files,
            cluster_test_files,
            label_map,
            config,
        )
    elif args.neuralnet:
        raise NotImplementedError(
            "Have not written the below yet custom to this script"
        )
        analyse_nn_feats(project_directory, label_map, config, args)
    else:
        raise ValueError("Should be neural net or manual")


def analyse_manual_feats(
    project_directory,
    cluster_train_files,
    cluster_test_files,
    label_map,
    config,
):
    """Analyse the features of the clusters manually extracted

    Args:
        project_directory (str): Location of the project directory
        cluster_train_files (list): List of the train files with
            clusters
        cluster_test_files (list): List of the test files with
            clusters
        label_map (dict): Map from the label name to number
        config (dict): Configuration for this script
    """

    # aggregate cluster features into collated df
    train_dfs = []
    test_dfs = []

    for index, file in enumerate(cluster_train_files):
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
        train_dfs.append(cluster_df)

    for index, file in enumerate(cluster_test_files):
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
        test_dfs.append(cluster_df)

    # aggregate dfs into one big df
    train_df = pl.concat(train_dfs)
    train_df = train_df.to_pandas()
    test_df = pl.concat(test_dfs)
    test_df = test_df.to_pandas()

    # check columns same
    assert train_df.columns == test_df.columns
    columns = train_df.columns

    # get features present in the dataframe
    not_features = ["clusterID", "x_mean", "y_mean", "type", "file_name"]
    features = [x for x in columns if x not in not_features]

    # now remove features not selected by user
    user_selected_features = config["features"]
    removed_features = [f for f in features if f not in user_selected_features]
    print("Removed features: ", removed_features)
    features = [f for f in features if f in user_selected_features]
    print("Features analysed: ", features)

    # feature vector
    train_X = train_df[features].values
    test_X = test_df[features].values

    # label vector
    train_unique_vals = sorted(train_df.type.unique())
    test_unique_vals = sorted(test_df.type.unique())
    labs = sorted(label_map.keys())
    assert labs == train_unique_vals
    assert labs == test_unique_vals
    train_Y = train_df.type.map(label_map).values
    test_Y = test_df.type.map(label_map).values

    # file names
    train_names = train_df.file_name
    test_names = test_df.file_name

    # train/test results init
    train_results = {"model": [], "f1": [], "acc": []}
    test_results = {"model": [], "f1": [], "acc": []}

    save_dir = os.path.join(project_directory, "output")

    # 4. Logistic regression
    if "log_reg" in config.keys():
        parameters = config["log_reg"]
        penalty = parameters["penalty"]
        C = parameters["C"]
        save_dir = os.path.join(project_directory, "output/log_reg")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model = LogisticRegression(max_iter=1000, penalty=penalty, C=C)
        train_results, test_results = gen_fn(
            model,
            "log_reg",
            train_X,
            train_Y,
            train_names,
            test_X,
            test_Y,
            test_names,
            train_results,
            test_results,
            label_map,
            save_dir,
        )

    # 5. Decision tree
    if "dec_tree" in config.keys():
        parameters = config["dec_tree"]
        max_depth = parameters["max_depth"]
        max_features = parameters["max_features"]
        save_dir = os.path.join(project_directory, "output/dec_tree")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
        train_results, test_results = gen_fn(
            model,
            "dec_tree",
            train_X,
            train_Y,
            train_names,
            test_X,
            test_Y,
            test_names,
            train_results,
            test_results,
            label_map,
            save_dir,
        )

    # 6. K-NN
    if "knn" in config.keys():
        parameters = config["knn"]
        n_neighbors = parameters["n_neighbors"]
        weights = parameters["weights"]
        save_dir = os.path.join(project_directory, "output/knn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        train_results, test_results = gen_fn(
            model,
            "knn",
            train_X,
            train_Y,
            train_names,
            test_X,
            test_Y,
            test_names,
            train_results,
            test_results,
            label_map,
            save_dir,
        )

    # 7. SVM
    if "svm" in config.keys():
        parameters = config["svm"]
        C = parameters["C"]
        kernel = parameters["kernel"]
        gamma = parameters["gamma"]
        save_dir = os.path.join(project_directory, "output/svm")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        train_results, test_results = gen_fn(
            model,
            "svm",
            train_X,
            train_Y,
            train_names,
            test_X,
            test_Y,
            test_names,
            train_results,
            test_results,
            label_map,
            save_dir,
        )

    # save results
    save_path = os.path.join(project_directory, "output/train_results.csv")
    df = pd.DataFrame(train_results)
    df.to_csv(save_path)

    save_path = os.path.join(project_directory, "output/test_results.csv")
    df = pd.DataFrame(test_results)
    df.to_csv(save_path)


def class_report_fn(df):
    """Produce class report for the given dataframe and indices

    Args:
        df (DataFrame): Contains the results

    Returns:
        conf_maxtrix (array): Confusion matrix
        f1 (float): F1 score
        acc (float): Accuracy score"""

    # take average prediction across all the clusters for each fov
    df = df.group_by("name").mean()
    # if average prediction is above 0.5 then predict as 1 otherwise 0
    df = df.with_columns(
        pl.when(pl.col("output") < 0.5).then(0).otherwise(1).alias("output")
    )

    # calculate classification report
    y_true = df["target"].to_list()
    y_pred = df["output"].to_list()
    print(classification_report(y_true, y_pred))

    print("Rows = True; Columns = Prediction")
    conf_matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    return conf_matrix, f1, acc


def gen_fn(
    model,
    model_name,
    train_X,
    train_Y,
    train_names,
    test_X,
    test_Y,
    test_names,
    train_results,
    test_results,
    label_map,
    save_dir,
):
    """

    Args:
        model (sklearn model): Model from sklearn to be trained
        model_name (string): Name of the model
        train_X (list): Input training features
        train_Y (list): Training labels
        train_names (list): Names of training files
        test_X (list): Input test features
        test_Y (list): Test labels
        test_names (list):  Names of test files
        train_results (dict): Dictionary to add training results to
        test_results (dict): Dictionary to add test results to
        label_map (dict): Map from integers to strings for labels
        save_dir (string): Where to save output

    Returns:
        train_results (dict): Dictionary of training results
        test_results (dict): Dictionary of test results

    """

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    # scale data
    scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # fit & predict
    model = model.fit(train_X)
    train_predict = model.predict(train_X)
    test_predict = model.predict(test_X)

    # prediction by the best model
    train_df_output = pl.DataFrame(
        {"name": train_names, "output": train_predict, "target": train_Y}
    )
    test_df_output = pl.DataFrame(
        {"name": test_names, "output": test_predict, "target": test_Y}
    )

    print(f"--- Classification report (train set)")
    train_confusion_matrix, f1_train, acc_train = class_report_fn(train_df_output)
    print(f"--- Classification report (test set)")
    test_confusion_matrix, f1_test, acc_test = class_report_fn(test_df_output)

    col_names = list(dict(sorted(label_map.items())).keys())

    # append results
    train_results["model"].append(model_name)
    train_results["f1"].append(f1_train)
    train_results["acc"].append(acc_train)
    test_results["model"].append(model_name)
    test_results["f1"].append(f1_test)
    test_results["acc"].append(acc_test)

    # save train results
    df_save = pd.DataFrame(train_confusion_matrix, columns=col_names, index=col_names)
    df_save_path = os.path.join(save_dir, f"{model_name}_confmatrix_train.csv")
    df_save.to_csv(df_save_path)

    # save test results
    df_save = pd.DataFrame(test_confusion_matrix, columns=col_names, index=col_names)
    df_save_path = os.path.join(save_dir, f"{model_name}_confmatrix_test.csv")
    df_save.to_csv(df_save_path)

    return train_results, test_results


if __name__ == "__main__":
    main()


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
    if not args.automatic:
        model_loc = os.path.join(
            project_directory, "models", f"fold_{fold}", model_name
        )
    elif args.automatic:
        model_dir = os.path.join(project_directory, "models", f"fold_{fold}")
        model_list = os.listdir(model_dir)
        assert len(model_list) == 1
        model_name = model_list[0]
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

    # label vector
    unique_vals = sorted(df.type.unique())
    labs = sorted(label_map.keys())
    assert labs == unique_vals
    data_labels = df.type.map(label_map).values

    # file names
    names = df.file_name

    # Analyses
    # 1. Plot boxplots of features
    df_save = df[features + ["type"]]
    df_save_path = os.path.join(project_directory, "output/features_nn.csv")
    df_save.to_csv(df_save_path, index=False)
    if config["boxplots"]:
        plot_boxplots(features, df)

    # ---------------------------------------------------------------------- #
    # Prediction methods taking in the folds
    # ---------------------------------------------------------------------- #

    X, Y, train_indices_main, val_indices_main, test_indices_main = prep_for_sklearn(
        data_feats, data_labels, names, args
    )

    print(np.array(X).shape)
    print(np.array(X))
    input_chan = config["nn_feat"][model_type]["ClusterEncoderChannels"][0][0]
    loc_chan = config["nn_feat"][model_type]["LocEncoderChannels_global"][-1][-1]
    with open("config/process.yaml", "r") as ymlfile:
        process_config = yaml.safe_load(ymlfile)
    cluster_features = process_config["cluster_feat"]
    print("cluster features", cluster_features, "length", len(cluster_features))
    print("loc chan", loc_chan)
    print("input channels - this should be sum of last two", input_chan)
    print(
        f"i think that first {loc_chan} is from nn and then the remaining {len(cluster_features)} is manual feature"
    )
    raise ValueError("check what to do")

    # train/test indices are list of lists
    # with one list for each fold
    train_indices_main = [train_indices_main[fold]]
    test_indices_main = [test_indices_main[fold]]

    # 2. Logistic regression
    if "log_reg" in config.keys():
        parameters = config["log_reg"]
        save_dir = os.path.join(project_directory, "output/log_reg_nn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
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
            fold,
            save_dir,
            label_map,
        )

    # 3. Decision tree
    if "dec_tree" in config.keys():
        parameters = config["dec_tree"]
        save_dir = os.path.join(project_directory, "output/dec_tree_nn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
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
            fold,
            save_dir,
            label_map,
        )

    # 4. K-NN
    if "knn" in config.keys():
        parameters = config["knn"]
        save_dir = os.path.join(project_directory, "output/knn_nn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_model = knn(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
            fold,
            save_dir,
            label_map,
        )

    # 5. SVM
    if "svm" in config.keys():
        parameters = config["svm"]
        save_dir = os.path.join(project_directory, "output/svm_nn")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_model = svm(
            X,
            Y,
            train_indices_main,
            val_indices_main,
            test_indices_main,
            parameters,
            names,
            args,
            fold,
            save_dir,
            label_map,
        )
