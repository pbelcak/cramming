#!/bin/bash

# set the project name env variable
export PROJECT_NAME=cramming

# cd into the project dir
cd /lustre/fsw/portfolios/nvr/users/pbelcak/${PROJECT_NAME}

# get the python version
python3 --version

echo "$SUBMIT_NAME"
echo "$SUBMIT_SAVE_ROOT"

# install the project requirements
python3 -m pip install .
