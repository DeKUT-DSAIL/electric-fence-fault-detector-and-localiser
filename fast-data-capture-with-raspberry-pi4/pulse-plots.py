#load and plot the saved sampled signal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

len_win = 200
t1 = 0
ns = 10000
fs = 31e6
t2 = ns / fs

filename = './data/test.csv'

t = np.linspace(t1, t2, num=ns) * 1000 #time in milliseconds
v = pd.read_csv(filename, header=None) #load the signal file


plt.figure(16,4)
plt.plot(t,v)
plt.axhline(color = 'black')
plt.xlabel('$Time\ (ms)$', fontsize=15)
plt.ylabel('$Voltage\ (V)$', fontsize=15)
plt.show()