#!/bin/bash

pip3 install virtualenv

python -m venv tdr-env

source tdr-env/bin/activate

pip3 install numpy
pip3 install pandas
pip3 install matplotlib
