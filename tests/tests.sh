#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

# Test preprocess
pytest tests/preprocess.py

# Activate correct environment
micromamba activate rapids=23.10

# Test feat extract
pytest tests/feat_extract.py

# Activate correct environment
micromamba activate locpix-points

# Test preprocess
pytest tests/process.py