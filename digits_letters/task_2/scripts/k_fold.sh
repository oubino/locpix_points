#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

# Test train
python scripts/k_fold.py

# remove files regardless of last script success
#echo "removing output end"
#rm -r semore_expts/output
