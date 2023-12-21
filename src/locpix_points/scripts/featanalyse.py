"""Feature analysis module

Module takes in the .parquet files and analyses features

Config file at top specifies the analyses we want to run
"""

import argparse
import json
import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import seaborn as sns
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
import yaml

config = {
    "pca_vs_convex_hull": False,
    "boxplots": False,
    "umap": False,
    "log_reg": True,
    "dec_tree": True,
    "svm": True,
    "knn": True,
}

label_map = {
    "fib": 0,
    "iso": 1,
}


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

    args = parser.parse_args(argv)

    project_directory = args.project_directory

    # load config
    # with open(args.config, "r") as ymlfile:
    #    config = yaml.safe_load(ymlfile)

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

    # feature vector
    data_feats = df[features].values
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
        for f in features:
            sns.boxplot(data=df, x=f, y="type")
            plt.show()

    # 3. Plot UMAP
    if config["umap"]:
        warnings.warn("Not generic code")
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data_feats_scaled)

        # Plot UMAP - normal vs cancer
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[sns.color_palette()[x] for x in df.type.map(umap_config)],
            label=[x for x in df.type.map({"normal": 0, "cancer": 1})],
        )
        normal_patch = mpatches.Patch(color=sns.color_palette()[0], label="Normal")
        cancer_patch = mpatches.Patch(color=sns.color_palette()[1], label="Cancer")
        plt.legend(handles=[normal_patch, cancer_patch])
        plt.gca().set_aspect("equal", "datalim")
        plt.title("UMAP projection of the dataset", fontsize=24)
        plt.show()

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
        plt.gca().set_aspect("equal", "datalim")
        plt.title("UMAP projection of the dataset", fontsize=24)
        plt.show()

    # ---------------------------------------------------------------------- #
    # Prediction methods taking in the folds
    # ---------------------------------------------------------------------- #

    warnings.warn(
        "There must be a config file for k_fold.yaml in the directory for this to work"
    )

    # load config
    config_path = os.path.join(args.project_directory, "config/k_fold.yaml")
    with open(config_path, "r") as ymlfile:
        k_fold_config = yaml.safe_load(ymlfile)

    splits = k_fold_config["splits"]
    train_folds = splits["train"]
    val_folds = splits["val"]
    test_folds = splits["test"]

    df = pl.DataFrame(
        {
            "X": data_feats_scaled,
            "Y": data_labels,
            "name": names,
        }
    )

    # get indices of train/test for CV
    train_indices_main = []
    test_indices_main = []

    for index, train_fold in enumerate(train_folds):
        val_fold = val_folds[index]
        test_fold = test_folds[index]

        train_bool = df["name"].is_in(train_fold).to_list()
        val_bool = df["name"].is_in(val_fold).to_list()
        test_bool = df["name"].is_in(test_fold).to_list()

        train_indices = np.where(train_bool)[0]
        val_indices = np.where(val_bool)[0]
        test_indices = np.where(test_bool)[0]

        train_indices = np.append(train_indices, val_indices)

        train_indices_main.append(train_indices)
        test_indices_main.append(val_indices)

        if any(i in train_indices for i in test_indices):
            raise ValueError("Should not share common values")

    num_features = len(df["X"][0])
    print("Num features: ", num_features)

    X = df["X"].to_list()
    Y = df["Y"].to_list()

    # 4. Logistic regression
    if config["log_reg"]:
        cv = iter(zip(train_indices_main, test_indices_main))

        model = LogisticRegression(max_iter=1000)
        parameters = {"penalty": ["l1", "l2"], "C": [0.1, 0.5, 1]}
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
        print(df.sort("rank_test_score"))

        best_model = clf.best_estimator_
        best_feats = dict(zip(features, best_model.coef_[0].tolist()))
        print(
            "Coeffs", sorted(best_feats.items(), key=lambda x: abs(x[1]), reverse=True)
        )

    # 5. Decision tree
    if config["dec_tree"]:
        cv = iter(zip(train_indices_main, test_indices_main))

        model = DecisionTreeClassifier()

        parameters = {
            "max_depth": [40, 45, 50],
            "max_features": [4, 5, 6],
        }
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
        print(df.sort("rank_test_score"))

        best_model = clf.best_estimator_
        print("length", best_model.feature_importances_.tolist())
        best_feats = dict(zip(features, best_model.feature_importances_.tolist()))
        print(
            "Coeffs", sorted(best_feats.items(), key=lambda x: abs(x[1]), reverse=True)
        )

    # 6. SVM
    if config["svm"]:
        cv = iter(zip(train_indices_main, test_indices_main))

        model = SVC()

        parameters = {
            "C": [1],
            #'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            "kernel": ["rbf"],
            "gamma": ["scale"],  # , 'auto']
        }

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
        print(df.sort("rank_test_score"))

    # 8. K-NN
    if config["knn"]:
        cv = iter(zip(train_indices_main, test_indices_main))

        model = KNeighborsClassifier(
            n_neighbors=3,
        )

        parameters = {"n_neighbors": [3], "weights": ["distance"]}

        clf = GridSearchCV(model, parameters, cv=cv)

        print("--------KNN---------")
        print("--------------------")

        clf.fit(X, Y)
        df = pl.DataFrame(clf.cv_results_)
        df = df.select(
            pl.col(
                [
                    "param_n_neighbors",
                    "param_weights",
                    "mean_test_score",
                    "std_test_score",
                    "rank_test_score",
                ]
            )
        )
        print(df.sort("rank_test_score"))

    # save yaml file
    # yaml_save_loc = os.path.join(project_directory, "featextract.yaml")
    # with open(yaml_save_loc, "w") as outfile:
    #    yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
