#!/bin/bash

# un-tar python and packages
tar -xzf python36.tar.gz
tar -xzf packages.tar.gz

# set path, pythonpath, and working directory as home
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

# run your script
python3 mll_calc.py $1 $2 $3 $4 $5 $6 $7 $8 $9 
