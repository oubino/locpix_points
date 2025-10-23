#! /usr/bin/bash

# Retrieve micromamba
source /root/micromamba/etc/profile.d/micromamba.sh

# Activate correct environment
micromamba activate locpix-points

while getopts ":a:bf:" opt; do
  case $opt in
    f)
      echo "line 12"
      fold="$OPTARG"
      echo "line 14"
      python scripts/k_fold.py -f $fold
      echo "line 16"
      exit 1
      ;;
  esac
done

# Test train
python scripts/k_fold.py

# remove files regardless of last script success
#echo "removing output end"
#rm -r semore_expts/output
