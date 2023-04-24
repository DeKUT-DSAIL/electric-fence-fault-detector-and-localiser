#!/bin/bash



root_path="./data/test"

mkdir -p $root_path

file_path="$root_path/test.csv"
sudo ./adc > $file_path

python signal-plot.py -f test


