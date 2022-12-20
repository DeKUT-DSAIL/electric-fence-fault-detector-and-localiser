#!/bin/bash

sudo ./adc > ./data/test.csv


source /home/pi/basic-env/bin/activate

python pulse-plots.py
