#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

# Preprocess
if pytest semore_expts/preprocess.py
then
    echo "preprocess successful"
else
    echo "removing output"
    rm -r semore_expts/output
    exit
fi

# Activate correct environment
micromamba activate rapids=23.10

# Test feat extract
if pytest semore_expts/featextract.py
then
    echo "feat extract successful"
else
    echo "removing output"
    rm -r semore_expts/output
    exit
fi

# Activate correct environment
micromamba activate locpix-points

# Test feat extract
if pytest semore_expts/process.py
then
    echo "processed successfuly"
else
    echo "removing output"
    rm -r semore_expts/output
    exit
fi

# Test train
if pytest semore_expts/k_fold.py
then
    echo "trained model"
else
    echo "removing output"
    rm -r semore_expts/output
    exit
fi

# remove files regardless of last script success
echo "removing output end"
rm -r semore_expts/output


