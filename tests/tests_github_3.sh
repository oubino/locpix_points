#! /usr/bin/bash

# Test k fold
if pytest tests/k_fold.py
then
    echo "k fold successful"
else
    echo "removing output"
    rm -r tests/output
    exit
fi

# Test feat analysis
if pytest tests/featanalysis.py
then
    echo "feat analysis successful"
else
    echo "removing output"
    rm -r tests/output
    exit
fi

# remove files regardless of last script success
echo "removing output end"
rm -r tests/output
