# This module contain functions needed for signal visualisation.


import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def path_generator(files_dir):
    """Generates paths to files
    Args: files_dir - files directory
    returns a list containing paths to files
    """

    paths = [os.path.join(files_dir, pulse_file).replace('\\', '/') for pulse_file in os.listdir(files_dir)]
    
    return paths 


def file_loader(paths):
    """Loads a random file
    Args: paths - list containing files' paths
    returns loaded signal
    """
    
    invalid_file = True
    
    while invalid_file:
        indx = np.random.randint(0, len(paths))
        file_path = paths[indx]

        v = np.array(list(map(float, pd.read_csv(file_path)['VC mem handle 5'].tolist()[1:-1])))

        if np.max(v) < 4 and np.min(v) > -4:
            #print('Valid file')
            invalid_file = False
            return v
        else:
            print('Invalid file. Searching for another file')
            

def graph_plot(x, y, s_indx, e_indx, plot_title, fig_size):
    """Plots graph of x against y
    Args: x- independent variable (time)
          y- dependent variable (voltage)
          plot_title- title of the plot
          x_label- x axis label
          y_label- y axis label
          fig_size- size of figure (float, float)
    """
    
    plt.figure(figsize=fig_size)
    plt.plot(x[s_indx: e_indx], y[s_indx: e_indx])
    plt.axhline(color = 'red', linewidth=2)
    plt.xlabel('$Time\ (ms)$', fontsize=15)
    plt.ylabel('$Voltage\ (V)$', fontsize=15)
    plt.title(plot_title, fontsize=15)
    plt.show()
    
    
          


def single_plot(v, t, plot_title):
    """Plots a single plot
    Args: v- an array of sampled values of signal
          t- an array of sampling time interval
          plot_title - title of the plot
    """

    fig_size = (10,5)
    x, y = t, v
    s_indx, e_indx = 0, len(v)
    graph_plot(x, y, s_indx, e_indx, plot_title, fig_size)

    
    
def edges_detection(v, t, n_edges):
    """Detects peaks in a signal
    Args: v- an array of sampled values of signal
          t- an array of sampling time interval
          n_edges - number of edges 
          returns a dictionary containing indices of n_peaks
          rising and n_peaks falling edges
    """
    
    if len(v) != len(t):
        print('Unmatching lengths of time and voltage arrays')

    gradients = np.diff(v)
    negative_peaks = np.argsort(gradients)[:n_edges].tolist()
    positive_peaks = np.argsort(gradients)[-n_edges:].tolist()

    edges_indices = {'rising_edges':positive_peaks,
             'falling_edges':negative_peaks}

    return edges_indices
    


def edges_plots(v, t, edges, win_size, prior_samples):
    """Plots the rising and falling edges of a signal
    Args: v- an array of sampled values of signal
          t- an array of sampling time interval
          edges- a dictinary of edges' indices
          win_size- window size
          prior_samples- number of samples prior the edge
    """
    
    x, y = t, v
    fig_size = mpl.rcParams['figure.figsize']
    for edge in edges:
        if 'rising' in edge:
            plot_title = 'Rising edge'
            for indx in edges[edge]:
                s_indx = indx - prior_samples #start index
                e_indx = indx + win_size #end index

                if s_indx < 0:
                    s_indx = 0

                if np.sum(v) < 5000: #short circuit signals
                    if np.max(v[s_indx: e_indx]) > 1: # filter out bus conflicted signals 
                        graph_plot(x, y, s_indx, e_indx, plot_title, fig_size)

                else: #open circuit signals
                    graph_plot(x, y, s_indx, e_indx, plot_title, fig_size)

        else:
            plot_title = 'Falling edge'
            for indx in edges[edge]:
                s_indx = indx - prior_samples #start index
                e_indx = indx + win_size #end index

                if s_indx < 0:
                    s_indx = 0

                if np.sum(v) < 5000: #short circuit signals
                    if np.min(v[s_indx: e_indx]) < -1:
                        graph_plot(x, y, s_indx, e_indx, plot_title, fig_size)

                else:
                    graph_plot(x, y, s_indx, e_indx, plot_title, fig_size)

                    
            
def signal_sum(paths):
    """Computes the sum sampled values.
    Args: paths- list containing files' paths
    returns a dictionary containing sum of sampled signals
    """
    
    sums = {}
    for file_path in paths:
        signal_sum = np.sum(np.array(list(map(float, pd.read_csv(file_path)['VC mem handle 5'].tolist()[1:-1]))))
        sums[file_path] = signal_sum
        
    return sums

     