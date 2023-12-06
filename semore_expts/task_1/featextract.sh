#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate rapids=23.10

# Test feat extract
python semore_expts/task_1/featextract.py
