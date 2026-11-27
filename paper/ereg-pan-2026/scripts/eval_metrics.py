"""
eval_metrics - Calculate and save model performance metrics.

Uses 0.5 probability threshold for binary classification.
Uses columns from input data:
    GT, simple_prob, NN_prob, Combined_prob
"""

import argparse

import pandas as pd
from pandas import Series
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calc_metrics(
    GTs: Series,
    probs: Series,
    pos_label: int,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate performance metrics and return a single-row dataframe.

    Args:
        pos_label (int):
            Binary label: 0 or 1
    """
    preds = probs > threshold
    preds = preds.astype(int)
    
    tn, fp, _, _ = confusion_matrix(  # tn, fp, fn, tp
        GTs, preds, labels=[int(not pos_label), pos_label]
    ).ravel()
    specificity = tn / (tn + fp)
    
    metrics = {
        "BA": balanced_accuracy_score(GTs, preds),
        "AUROC": roc_auc_score(GTs, probs),
        "sensitivity": recall_score(GTs, preds, pos_label=pos_label),
        "specificity": specificity,
        "precision": precision_score(GTs, preds, pos_label=pos_label)
    }

    return metrics
    

# Get input and output locations
parser = argparse.ArgumentParser(
    description="Calculate and save model performance metrics."
)
parser.add_argument(
    "-i",
    "--input_path",
    action="store",
    type=str,
    required=True,
    help="Path to input file with predictions."
)
parser.add_argument(
    "-o",
    "--output_path",
    action="store",
    type=str,
    required=True,
    help="Destination path for output file containing metrics."
)
args = parser.parse_args()

input_table = pd.read_csv(args.input_path)

GTs = input_table["GT"]

# Simple prediction model (e.g. logistic regression)
simple_probs = input_table["simple_prob"]
simple_metrics_pos0 = calc_metrics(GTs, simple_probs, pos_label=0)
simple_metrics_pos0["Model"] = "Simple, positive class: 0"
simple_metrics_pos1 = calc_metrics(GTs, simple_probs, pos_label=1)
simple_metrics_pos1["Model"] = "Simple, positive class: 1"

# NN prediction model based on ClusterNet
nn_probs = input_table["NN_prob"]
nn_metrics_pos0 = calc_metrics(GTs, nn_probs, pos_label=0)
nn_metrics_pos0["Model"] = "NN, positive class: 0"
nn_metrics_pos1 = calc_metrics(GTs, nn_probs, pos_label=1)
nn_metrics_pos1["Model"] = "NN, positive class: 1"

# Combined probabilities
comb_probs = input_table["Combined_prob"]
comb_metrics_pos0 = calc_metrics(GTs, comb_probs, pos_label=0)
comb_metrics_pos0["Model"] = "Combined, positive class: 0"
comb_metrics_pos1 = calc_metrics(GTs, comb_probs, pos_label=1)
comb_metrics_pos1["Model"] = "Combined, positive class: 1"

df = pd.DataFrame(
    data=[
        simple_metrics_pos0,
        simple_metrics_pos1,
        nn_metrics_pos0,
        nn_metrics_pos1,
        comb_metrics_pos0,
        comb_metrics_pos1
    ]
)

df.to_csv(args.output_path)
