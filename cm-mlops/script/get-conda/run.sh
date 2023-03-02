#!/bin/bash

if ! command -v conda &> /dev/null
then
    echo "conda command not found"
    echo "This script requires Anaconda or Miniconda to be installed."
    echo "Installing Anaconda..."
    bash Anaconda-latest-Linux-x86_64.sh
else
   echo "conda found!"
fi

echo "conda environment: ${CM_ENV_CONDA}"
echo "python environment: ${CM_ENV_PYTHON}"
conda create --name ${CM_ENV_CONDA} python=${CM_ENV_PYTHON} -y
eval "$(conda shell.bash hook)"
#conda activate ${CM_ENV_CONDA}

