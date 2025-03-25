#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

# Run k_fold split
python scripts/generate_k_fold_splits.py
