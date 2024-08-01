#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate rapids=23.10

# Test feat extract
python scripts/featextract_train.py
python scripts/featextract_test.py

