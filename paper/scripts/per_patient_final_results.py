# Calculate final per-patient results

import polars as pl
import polars.selectors as cs
from sklearn.metrics import confusion_matrix

# load in k_fold_df.csv
k_fold_df = "output/combined_classification/k_fold_df.csv"
k_fold_df = pl.read_csv(k_fold_df)

# calculate per-patient confusion matrix
k_fold_df = k_fold_df.drop(k_fold_df.columns[0]).drop("File")
agg_df = k_fold_df.group_by(pl.col(["patient","GT"])).agg(pl.col(["simple_prob", "NN_prob", "Combined_prob"]).mean()).sort("patient")
x = agg_df.select(pl.col(
    ["GT", "Combined_prob"]
)).to_numpy()

y_true = x[:,0]
y_prob = x[:,1]
y_pred = [1 if x >= 0.5 else 0 for x in y_prob]

print("dSTORM predictions")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# only high EREG patients
linked_files = "config/linked_files.csv"
linked_files = pl.read_csv(linked_files)
linked_files = linked_files.drop_nulls()

high_ereg_patients = linked_files.filter(
    pl.col("%pos_ereg") >= 50.0
)
high_ereg_patients = high_ereg_patients.with_columns(
    pl.col("patient").str.strip_prefix("PAT").cast(pl.Int64)
)

df = high_ereg_patients.join(agg_df, on="patient")
df = df.drop("GT")

x = df.select(pl.col(
    ["response", "Combined_prob"]
)).to_numpy()

y_true = x[:,0]
y_prob = x[:,1]
y_pred = [1 if x >= 0.5 else 0 for x in y_prob]

print("dSTORM predictions (only high EREG)")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# IHC prediction (all patients)
x = linked_files.select(pl.col(
    ["response", "%pos_ereg"]
)).to_numpy()

y_true = x[:,0]
y_prob = x[:,1]
y_pred = [1 if x >= 50.0 else 0 for x in y_prob]

print("IHC predictions")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# IHC prediction (high EREG patients)
x = high_ereg_patients.select(pl.col(
    ["response", "%pos_ereg"]
)).to_numpy()

y_true = x[:,0]
y_prob = x[:,1]
y_pred = [1 if x >= 50.0 else 0 for x in y_prob]

print("IHC predictions (only high EREG)")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# dSTORM + IHC prediction
low_ereg_patients = linked_files.filter(
    pl.col("%pos_ereg") < 50.0
)
low_ereg_patients_list = low_ereg_patients.select(
    pl.col("patient")
).to_series().to_list()
low_ereg_patients_pred = [0] * len(low_ereg_patients)
low_ereg_patients_gt = low_ereg_patients.select(
    pl.col("response")
).to_series().to_list()

df = high_ereg_patients.join(agg_df, on="patient")
df = df.drop("GT")

high_ereg_patients_list = df.select(
    pl.col("patient")
).to_series().to_list()
high_ereg_patients_list = [f"PAT{x}" for x in high_ereg_patients_list]
high_ereg_patients_pred = df.select(
    pl.col("Combined_prob")
).to_series().to_list()
high_ereg_patients_pred = [1 if x >= 0.5 else 0 for x in high_ereg_patients_pred]
high_ereg_patients_gt = df.select(
    pl.col("response")
).to_series().to_list()

y_true = low_ereg_patients_gt + high_ereg_patients_gt
y_pred = low_ereg_patients_pred + high_ereg_patients_pred
pats = low_ereg_patients_list + high_ereg_patients_list

print("Final prediction from joined pipeline")
print(pats)
print(y_pred)
cm = confusion_matrix(y_true, y_pred)
print(cm)