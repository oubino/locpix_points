"""
simple_classn_and_comb_cells.py - predictions from logistic regression and combination with neural network results per cell
"""

import os  # noqa: I001

import numpy as np
import polars as pl
import yaml
# from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import class_likelihood_ratios, roc_auc_score
from sklearn.preprocessing import StandardScaler

from functions import extract_features

balanced_accuracy = True

if balanced_accuracy:
    from sklearn.metrics import balanced_accuracy_score as accuracy_score
else:
    from sklearn.metrics import accuracy_score


def feat_extract(features, feature_str, file_filter):
    df_filtered = features.filter(pl.col("file").is_in(file_filter))
    # order in same order as file filter
    # map value to position
    order_map = {v: i for i, v in enumerate(file_filter)}
    
    df_filtered = (
        df_filtered.with_columns(
            pl.col("file").map_dict(order_map).alias("_order")
        )
        .sort("_order")
        .drop("_order")
    )
    return df_filtered["file"], df_filtered[feature_str].to_numpy(), df_filtered["gt"].to_numpy()

def feat_extract_overall(features, feature_list, train_folds, val_folds, test_folds):
    train_list = []
    train_list_gt = []

    val_list = []
    val_list_gt = []

    test_list = []
    test_list_gt = []

    for fold in range(5):
        train_files = train_folds[fold]
        val_files = val_folds[fold]
        test_files = test_folds[fold]

        new_train_files_list, X_train, Y_train = feat_extract(features, feature_list, train_files)
        new_val_files_list, X_val, Y_val = feat_extract(features, feature_list, val_files)
        new_test_files_list, X_test, Y_test = feat_extract(features, feature_list, test_files)

        assert (train_files == new_train_files_list).all()
        assert (val_files == new_val_files_list).all()
        assert (test_files == new_test_files_list).all()

        train_list.append(X_train)
        train_list_gt.append(Y_train)

        val_list.append(X_val)
        val_list_gt.append(Y_val)

        test_list.append(X_test)
        test_list_gt.append(Y_test)

    return train_list, train_list_gt, val_list, val_list_gt, test_list, test_list_gt


def simple_classify(model, X_train, Y_train, X_test, Y_test, train_scores, final_probs, final_preds, test_scores_auroc, test_scores_acc, test_scores_plr, feature_names):

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, Y_train)

    train_probs = model.predict_proba(X_train)[:,1]
    train_scores.append(roc_auc_score(Y_train, train_probs))

    test_probs = model.predict_proba(X_test)[:,1]
    final_probs.extend(test_probs)
    test_preds = model.predict(X_test)
    final_preds.extend(test_preds)

    test_scores_auroc.append(roc_auc_score(Y_test, test_probs))
    test_scores_acc.append(accuracy_score(Y_test, test_preds))
    test_scores_plr.append(class_likelihood_ratios(Y_test, test_preds)[0])

    return test_probs


def log_regr():
    """
    Do logistic regression...

    Using c=100, l1_ratio= .25, max_iter= 5000 penalty=elasticnet,
    found in nested validation.
    """
    # Preprocessing
    input_folder = "../cells/gt_label"
    files = os.listdir(input_folder)
    # channels = ["ereg"]

    # Class names
    # class_0_str = "no-response"
    # class_1_str = "any-response"

    # load in splits
    k_fold_file = "../output/config/k_fold.yaml"

    # load yaml
    with open(k_fold_file, "r") as ymlfile:
        k_fold_file = yaml.safe_load(ymlfile)

    train_folds = k_fold_file["splits"]["train"]
    val_folds = k_fold_file["splits"]["val"]
    test_folds = k_fold_file["splits"]["test"]

    # Get model features and store if necessary
    if not os.path.exists("../output/simple_classification"):
        os.makedirs("../output/simple_classification")

    features_path = "../output/simple_classification/features.parquet"

    if not os.path.exists(features_path):
        features = extract_features(files, input_folder)
        features.write_parquet(features_path)
    else:
        features = pl.read_parquet(features_path)

    features_list = [
        "cell_rgyration",
        "cell_linearity",
        "cell_planarity",
        "cell_length",
        "cell_area",
        "cell_perimeter",
    ]

    X_train_list, Y_train_list, X_val_list, Y_val_list, X_test_list, Y_test_list = (
        feat_extract_overall(features, features_list, train_folds, val_folds, test_folds)
    )

    # for split in splits
    train_scores_simple = []

    test_scores_simple_auroc = []
    test_scores_simple_acc = []
    test_scores_simple_plr = []

    final_test_files = []

    final_preds_simple = []

    # simple_feat_importances_list = []

    final_probs_simple = []

    for fold in range(5):
        for file in test_folds[fold]:
            # evaluate on test set
            final_test_files.append(file)

        X_train = X_train_list[fold]
        Y_train = Y_train_list[fold]

        X_val = X_val_list[fold]
        Y_val = Y_val_list[fold]

        X_train = np.vstack((X_train, X_val))
        Y_train = np.concatenate((Y_train, Y_val))

        X_test = X_test_list[fold]
        Y_test = Y_test_list[fold]

        # simple regression model
        model = LogisticRegression(
            C=100, l1_ratio=0.25, max_iter=5000, penalty="elasticnet", solver="saga"
        )

        # test_probs =
        simple_classify(
            model,
            X_train,
            Y_train,
            X_test,
            Y_test,
            train_scores_simple,
            final_probs_simple,
            final_preds_simple,
            test_scores_simple_auroc,
            test_scores_simple_acc,
            test_scores_simple_plr,
            features_list,
        )

    print("Mean AUROC on test set")
    print(np.mean(test_scores_simple_auroc))

    print("Mean acc on test set")
    print(np.mean(test_scores_simple_acc))

    return test_folds, final_test_files, final_probs_simple
