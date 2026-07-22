# Contains functions necessary for simple classification

## Includes additional stuff that is not required

from locpix_points.preprocessing.featextract import convex_hull, pca_cluster, basic_cluster_feats, convex_hull_cluster, pca_fn
from sklearn.cluster import DBSCAN
import polars as pl
from locpix_points.preprocessing import datastruc
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import class_likelihood_ratios, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def cluster_algo(df, eps, min_samples):
    arr = df[["x", "y"]].to_numpy()
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(arr)

    df = df.with_columns(
            pl.lit(clustering.labels_.astype("int32")).alias("clusterID")
    )
    
    # drop unclustered points
    df = df.filter(pl.col("clusterID") != -1)

    # Dropping all clusters with 2 or fewer locs - otherwise convex hull fails
    small_clusters = df.group_by("clusterID").agg(pl.count().alias("len")).filter(pl.col("len") < 3)
    df = df.filter(~pl.col("clusterID").is_in(small_clusters["clusterID"]))

    # remap the clusterIDs
    unique_clusters = list(df["clusterID"].unique())
    map = {value: i for i, value in enumerate(unique_clusters)}
    df = df.with_columns(pl.col("clusterID").map_dict(map).alias("clusterID"))

    return df

def cluster_feat_extract(file, input_folder, eps, min_samples):
    
    # load items
    item = datastruc.item(None, None, None, None, None)
    item.load_from_parquet(os.path.join(input_folder, file))
    df = item.df

    df = cluster_algo(df, eps, min_samples)

    # basic features (com cluster, locs per cluster, radius of gyration)
    basic_cluster_df = basic_cluster_feats(df)

    # pca on cluster (linearity, circularity see DIMENSIONALITY BASED SCALE SELECTION IN 3D LIDAR POINT CLOUDS)
    pca_cluster_df = pca_cluster(df)

    # convex hull (perimeter, area, length)
    convex_hull_cluster_df = convex_hull_cluster(df)

    # merge cluster df
    cluster_df = basic_cluster_df.join(pca_cluster_df, on="clusterID", how="inner")
    cluster_df = cluster_df.join(
        convex_hull_cluster_df, on="clusterID", how="inner"
    )

    # don't allow fewer than 3 clusters
    num_clusters = cluster_df["clusterID"].max()
    if num_clusters < 3:
        raise ValueError("2 or fewer clusters")

    # cluster density do this here
    cluster_df = cluster_df.with_columns(
        (pl.col("count") / pl.col("area_convex_hull")).alias("density_convex_hull")
    )
    cluster_df = cluster_df.with_columns(
        (pl.col("count") / pl.col("area_pca")).alias("density_pca")
    )

    cluster_df = cluster_df.rename(
        {"count": "locs_per_cluster"}
    )

    series = pl.Series("gt", [item.gt_label]*len(cluster_df))
    cluster_df = cluster_df.with_columns(
        series
    )

    return cluster_df

def rgyration(df):

    x_mean = df["x"].mean()
    y_mean = df["y"].mean()

    df = df.with_columns(
        ((pl.col("x") - x_mean) ** 2 + (pl.col("y") - y_mean) ** 2)
        .alias("squared_diff")
        )
    
    rgyration2 = df["squared_diff"].mean()    

    return rgyration2

def shape(df):

    arr = df[["x", "y"]].to_numpy()

    linearity, planarity, length, area = pca_fn(arr, dim=2)

    perimeter, _, _ = convex_hull(arr)

    return linearity, planarity, length, area, perimeter

def cell_feat_extract(file, input_folder):

    # load items
    item = datastruc.item(None, None, None, None, None)
    item.load_from_parquet(os.path.join(input_folder, file))
    df = item.df

    # extract features
    cell_linearity, cell_planarity, cell_length, cell_area, membrane_perimeter = shape(df)
    cell_rgyration = rgyration(df)


    cell_feat_df = pl.DataFrame(
        {"cell_rgyration": cell_rgyration,
         "cell_linearity": cell_linearity,
         "cell_planarity": cell_planarity,
         "cell_length": cell_length,
         "cell_area": cell_area,
         "cell_perimeter": membrane_perimeter,
         "gt": item.gt_label,
        }
    )

    return cell_feat_df


def extract_features(files, input_folder):

    cell_dfs = []

    for idx, file in enumerate(files):
        
        cell_df = cell_feat_extract(file, input_folder)
        cell_dfs.append(cell_df)

        if (idx+1) % 5: 
            print(f"{100*(idx+1)/len(files):.0f}% done")
    
    cell_df = pl.concat(cell_dfs)

    series = pl.Series("file", files)
    cell_df = cell_df.with_columns(
        series,
    )
    
    return cell_df

def print_results(test_files, predictions, train_scores, test_scores_auroc, test_scores_acc, test_scores_plr, feat_importances_names, feat_importances):
    print("Test files")
    print(test_files)

    print("--- Predictions ---")
    print(predictions)

    print("Train AUROC")
    print(train_scores)
    print(np.mean(train_scores), " +- ", np.std(train_scores, ddof=1))

    print("Test AUROC")
    print(test_scores_auroc)
    print(np.mean(test_scores_auroc), " +- ", np.std(test_scores_auroc, ddof=1))


    print("Test acc")
    print(test_scores_acc)
    print(np.mean(test_scores_acc), " +- ", np.std(test_scores_acc, ddof=1))

    print("Test plr")
    print(test_scores_plr)
    print(np.mean(test_scores_plr), " +- ", np.std(test_scores_plr, ddof=1))

    print("Feat importances")
    print(feat_importances_names)
    print(feat_importances)

def fit_and_predict(pipeline, X_train, Y_train, X_test, Y_test, train_scores, test_scores_auroc, test_scores_acc, test_scores_plr, final_preds, balanced_accuracy=False):
        
    pipeline.fit(X_train, Y_train)

    probs = pipeline.predict_proba(X_train)[:,1]
    train_scores.append(roc_auc_score(Y_train, probs))

    probs = pipeline.predict_proba(X_test)[:,1]
    preds = pipeline.predict(X_test)
    final_preds.extend(preds)

    test_scores_auroc.append(roc_auc_score(Y_test, probs))
    if balanced_accuracy:
        test_scores_acc.append(balanced_accuracy_score(Y_test, preds))
    else:
        test_scores_acc.append(accuracy_score(Y_test, preds))
    test_scores_plr.append(class_likelihood_ratios(Y_test, preds)[0])

    try:
        return np.array(pipeline.named_steps["logistic_regression"].coef_), probs, Y_test
    except:
        KeyError("Not log")

    try:
        return np.expand_dims(np.array(pipeline.named_steps["random_forest"].feature_importances_), axis=0), probs, Y_test
    except:
        KeyError("Not random forest")

def agg_cell(files, cluster_features, probs):

    cluster_features_preds = cluster_features.filter(pl.col("file").is_in(files)).with_columns(
        pl.lit(probs).alias("prob")
    )
    cluster_features_preds.group_by(
        "file", maintain_order=True,
    ).mean()

    Y = cluster_features_preds["gt"]
    probs = cluster_features_preds["prob"]
    preds = cluster_features_preds.with_columns(
        pl.when(pl.col("prob") > 0.5).then(1).otherwise(0).alias("pred")
    )
    preds = preds["pred"]

    return probs, preds, Y

def tune(X_train_list, Y_train_list, X_val_list, Y_val_list, param_list, pipeline_choice=None, balanced_accuracy=False):
    results_auroc = []
    results_acc = []
    for param in param_list:
        if pipeline_choice == "log":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),     
                ('logistic_regression', LogisticRegression(**param, solver="saga"))
            ])
        elif pipeline_choice == "rf":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),     
                ('random_forest', RandomForestClassifier(**param))
            ])
        results_auroc_ = []
        results_acc_ = []
        for fold in range(5):

            
            X_train = X_train_list[fold]
            Y_train = Y_train_list[fold]

            X_val = X_val_list[fold]
            Y_val = Y_val_list[fold]
           
            pipeline.fit(X_train, Y_train)
            probs = pipeline.predict_proba(X_val)[:,1]
            preds = pipeline.predict(X_val)
            results_auroc_.append(roc_auc_score(Y_val, probs))
            if balanced_accuracy:
                results_acc_.append(balanced_accuracy_score(Y_val, preds))
            else:
                results_acc_.append(accuracy_score(Y_val, preds))
            
        results_auroc.append(np.mean(results_auroc_))
        results_acc.append(np.mean(results_acc_))

    a, b, c = zip(*sorted(zip(results_auroc, results_acc, param_list), key=lambda x: x[0], reverse=True))
    print("AUROC results")
    print(a[0:3])
    print("Accuracy")
    print(b[0:3])
    print("Param list")
    print(c[0:3])

def feat_extract(features, feature_str, file_filter):
    df_filtered = features.filter(pl.col("file").is_in(file_filter))
    return df_filtered[feature_str].to_numpy(), df_filtered["gt"].to_numpy()

def feat_extract_overall(features, feature_list, train_folds, val_folds, test_folds):
    train_list = []
    train_list_gt = []

    val_list = []
    val_list_gt = []

    test_list = []
    test_list_gt = []

    for fold in range(len(train_folds)):
        train_files = train_folds[fold]
        val_files = val_folds[fold]
        test_files = test_folds[fold]

        X_train, Y_train = feat_extract(features, feature_list, train_files)
        X_val, Y_val = feat_extract(features, feature_list, val_files)
        X_test, Y_test = feat_extract(features, feature_list, test_files)

        train_list.append(X_train)
        train_list_gt.append(Y_train)

        val_list.append(X_val)
        val_list_gt.append(Y_val)

        test_list.append(X_test)
        test_list_gt.append(Y_test)

    return train_list, train_list_gt, val_list, val_list_gt, test_list, test_list_gt

def cell_predict_final(cluster_features, train_scores_cells, test_scores_cells_auroc, test_scores_cells_acc, test_scores_cells_plr, final_preds, fold, pipeline, X_train, Y_train, X_test, train_folds, val_folds, test_folds, balanced_accuracy=False):
    
    pipeline.fit(X_train, Y_train)
    
    train_probs = pipeline.predict_proba(X_train)[:,1]
    train_files = train_folds[fold] + val_folds[fold]
    train_probs, _, Y_train_cell = agg_cell(train_files, cluster_features, train_probs)
    train_scores_cells.append(roc_auc_score(Y_train_cell, train_probs))
    
    test_probs = pipeline.predict_proba(X_test)[:,1]

    test_probs, test_preds, Y_test_cell = agg_cell(test_folds[fold], cluster_features, test_probs)
    test_scores_cells_auroc.append(roc_auc_score(Y_test_cell, test_probs))
    if balanced_accuracy:
        test_scores_cells_acc.append(balanced_accuracy_score(Y_test_cell, test_preds))
    else:
        test_scores_cells_acc.append(accuracy_score(Y_test_cell, test_preds))
    test_scores_cells_plr.append(class_likelihood_ratios(Y_test_cell, test_preds)[0])
    final_preds.extend(test_preds)

    try:
        return np.array(pipeline.named_steps["logistic_regression"].coef_)
    except:
        KeyError("Not log")

    try:
        return np.expand_dims(np.array(pipeline.named_steps["random_forest"].feature_importances_), axis=0)
    except:
        KeyError("Not random forest")

def extract_importances(feature_names, feat_importances_list):
    feat_importances = np.vstack(feat_importances_list)
    feat_importances = np.mean(feat_importances, axis=0)
    feat_importances, feat_importances_names = zip(*sorted(zip(feat_importances, feature_names), key=lambda x: abs(x[0]), reverse=True))
    return feat_importances,feat_importances_names
