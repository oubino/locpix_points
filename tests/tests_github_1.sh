#! /usr/bin/bash

# make directory
mkdir tests/output

# Test preprocess
if pytest tests/preprocess.py
then
    echo "preprocess successful"
else
    echo "removing output"
    rm -r tests/output
    exit
fi