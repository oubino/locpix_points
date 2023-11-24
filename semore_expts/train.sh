#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

# Test train
if pytest semore_expts/k_fold.py
then
    echo "trained model"
else
    echo "removing output"
    rm -r semore_expts/output/processed
    rm semore_expts/output/k_fold.yaml
    rm semore_expts/output/train.yaml
    exit
fi

# remove files regardless of last script success
#echo "removing output end"
#rm -r semore_expts/output


