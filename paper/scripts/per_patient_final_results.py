# %% [markdown]
# # Calculate final per-patient results

# %% [markdown]
# ## Imports

# %%
import polars as pl
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

# %% [markdown]
# ## Load data in

# %%
# Load in k_fold_df.csv - per cell dSTORM predictions
dstorm_cells_df = "output/combined_classification/k_fold_df.csv"
dstorm_cells_df = pl.read_csv(dstorm_cells_df)

# IHC data
linked_files_df = "config/linked_files.csv"
linked_files_df = pl.read_csv(linked_files_df)
linked_files_df = linked_files_df.with_columns(
    pl.col("patient").str.strip_prefix("PAT").cast(pl.Int64)
)

# %% [markdown]
# ## Merge data, and add binary predictions and correctness

# %%
dstorm_pat_df = dstorm_cells_df.group_by(
    pl.col(["patient","GT"])
    ).agg(pl.col(["simple_prob", "NN_prob", "Combined_prob"]).mean()).sort("patient")
dstorm_pat_df = dstorm_pat_df.with_columns(
    (pl.col("Combined_prob") >= 0.5).cast(pl.Int64).alias("dSTORM_pred")
)
dstorm_pat_df = dstorm_pat_df.with_columns(
    (pl.col("dSTORM_pred") == pl.col("GT")).alias("dSTORM_correct")
)
dstorm_pat_df

# %%
linked_files_df = linked_files_df.with_columns(
    ((pl.col("%pos_ereg") >= 50) | (pl.col("%pos_areg") >= 50)).cast(pl.Int64).alias("IHC_pred")
)
linked_files_df = linked_files_df.with_columns(
    (pl.col("IHC_pred") == pl.col("response")).alias("IHC_correct")
)
linked_files_df

# %%
all_df = linked_files_df.join(dstorm_pat_df, on="patient")
all_df

# %% [markdown]
# ## Confusion matrices, all EREG levels by IHC

# %%
cm_storm_all = confusion_matrix(all_df["GT"].to_numpy(), all_df["dSTORM_pred"].to_numpy())
print()
print("_dSTORM confusion matrix, all samples_")
print("\tPred 0 | Pred 1")
print(f"Response 0 {cm_storm_all[0]}")
print(f"Response 1 {cm_storm_all[1]}")

# %% [markdown]
# ### When using IHC results, drop samples with no IHC results

# %%
has_IHC_df = all_df.drop_nulls()
has_IHC_df

# %%
cm_ihc_all = confusion_matrix(has_IHC_df["GT"].to_numpy(), has_IHC_df["IHC_pred"].to_numpy())
print()
print("_IHC confusion matrix, all samples_")
print("\tPred 0 | Pred 1")
print(f"Response 0 {cm_ihc_all[0]}")
print(f"Response 1 {cm_ihc_all[1]}")

# %% [markdown]
# ## McNemar's test, all EREG levels by IHC

# %%
n01 = (~has_IHC_df["dSTORM_correct"] & has_IHC_df["IHC_correct"]).sum()
n10 = (has_IHC_df["dSTORM_correct"] & ~has_IHC_df["IHC_correct"]).sum()
n_discordant = n01 + n10

print()
print("_Statistics, all samples with IHC results_")
print(f"Discordant counts: n01 (IHC better) = {n01}, n10 (dSTORM better) = {n10}, total = {n_discordant}")

if n_discordant == 0:
    print("No discordant pairs. The methods have identical outcomes vs GT (p = 1.0).")
else:
    # One-sided exact McNemar: "Is dSTORM + ML more accurate than IHC?"
    p_ml_greater = binomtest(k=n10, n=n_discordant, p=0.5, alternative='greater').pvalue
    print(f"Exact McNemar (one-sided, dSTORM > IHC): p = {p_ml_greater:.6g}")

# %% [markdown]
# ## Confusion matrices, only high EREG levels by IHC

# %%
high_ereg_pats_df = has_IHC_df.filter((pl.col("%pos_ereg") >= 50.0) | (pl.col("%pos_areg") >= 50.0))
high_ereg_pats_df

# %%
cm_storm_highereg = confusion_matrix(high_ereg_pats_df["GT"].to_numpy(), high_ereg_pats_df["dSTORM_pred"].to_numpy())
print()
print("_dSTORM confusion matrix, high-EREG samples_")
print("\tPred 0 | Pred 1")
print(f"Response 0 {cm_storm_highereg[0]}")
print(f"Response 1 {cm_storm_highereg[1]}")

# %%
cm_ihc_highereg = confusion_matrix(high_ereg_pats_df["GT"].to_numpy(), high_ereg_pats_df["IHC_pred"].to_numpy())
print()
print("_IHC confusion matrix, high-EREG samples_")
print("\tPred 0 | Pred 1")
print(f"Response 0 {cm_ihc_highereg[0]}")
print(f"Response 1 {cm_ihc_highereg[1]}")

# %% [markdown]
# ## McNemar's test, high EREG level by IHC

# %%
n01 = (~high_ereg_pats_df["dSTORM_correct"] & high_ereg_pats_df["IHC_correct"]).sum()
n10 = (high_ereg_pats_df["dSTORM_correct"] & ~high_ereg_pats_df["IHC_correct"]).sum()
n_discordant = n01 + n10

print()
print("_Statistics, samples with high EREG/AREG by IHC_")
print(f"Discordant counts: n01 (IHC better) = {n01}, n10 (dSTORM better) = {n10}, total = {n_discordant}")

if n_discordant == 0:
    print("No discordant pairs. The methods have identical outcomes vs GT (p = 1.0).")
else:
    # One-sided exact McNemar: "Is dSTORM + ML more accurate than IHC?"
    p_ml_greater = binomtest(k=n10, n=n_discordant, p=0.5, alternative='greater').pvalue
    print(f"Exact McNemar (one-sided, dSTORM > IHC): p = {p_ml_greater:.6g}")
    print()

# %%



