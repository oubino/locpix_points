#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

# Preprocess
python scripts/process.py
