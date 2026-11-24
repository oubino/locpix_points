# %% [markdown]
# # Calculate final per-patient results
# ## Run from paper/ereg-pan-2026 directory for valid save location
# with python scripts/per_patient_preds.py

# %% [markdown]
# ## Imports

# %%
import polars as pl

# %% [markdown]
# ## Load data in

# %%
# Load in k_fold_df.csv - per cell dSTORM predictions
dstorm_cells_df = "output/combined_classification/k_fold_df.csv"
dstorm_cells_df = pl.read_csv(dstorm_cells_df)

# %%
## Aggregate by patient and add predictions
dstorm_pat_df = dstorm_cells_df.group_by(
    pl.col(["patient","GT"])
    ).agg(pl.col(["simple_prob", "NN_prob", "Combined_prob"]).mean()).sort("patient")
dstorm_pat_df = dstorm_pat_df.with_columns(
    (pl.col("Combined_prob") >= 0.5).cast(pl.Int64).alias("dSTORM_pred")
)
dstorm_pat_df = dstorm_pat_df.with_columns(
    (pl.col("dSTORM_pred") == pl.col("GT")).alias("dSTORM_correct")
)

dstorm_pat_df.write_csv("output/combined_classification/dstorm_patient_preds.csv")

print(dstorm_pat_df)



