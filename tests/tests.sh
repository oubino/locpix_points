#! /usr/bin/bash

# make directory
mkdir tests/output

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

# Test preprocess

if pytest tests/preprocess.py
then
    echo "preprocess successful"
else
    echo "removing output"
    rm -r tests/output
    exit
fi

# Activate correct environment
micromamba activate rapids=23.10

# Test feat extract
if pytest tests/featextract.py
then
    echo "feat extract successful"
else
    echo "removing output"
    rm -r tests/output
    exit
fi

# Activate correct environment
micromamba activate locpix-points

# Test train
if pytest tests/train.py
then
    echo "trained model"
else
    echo "removing output"
    rm -r tests/output
    exit
fi

# remove files regardless of last script success
#echo "removing output end"
#rm -r tests/output


