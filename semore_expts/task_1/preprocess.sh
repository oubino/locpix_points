#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

# Preprocess
python semore_expts/task_1/preprocess.py



