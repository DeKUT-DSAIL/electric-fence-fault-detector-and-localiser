import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Plot sampled signals in a given directory')
parser.add_argument('-f',
                    '--folder',
                    type=str,
                    required=True,
                    metavar='',
                    help='Name of folder')
args = parser.parse_args()


len_win = 200
t1 = 0
ns = 10000
fs = 31e6
t2 = ns / fs




files_path = os.path.join('./data', args.folder)

t = np.linspace(t1, t2, num=ns) * 1000


for file in os.listdir(files_path):
    file_path = os.path.join(files_path, file)
    v = pd.read_csv(file_path, header=None)
    plt.plot(t,v)
    plt.title(file)
    plt.axhline(color = 'black')
    plt.xlabel('$Time\ (ms)$', fontsize=15)
    plt.ylabel('$Voltage\ (V)$', fontsize=15)
    plt.show()
