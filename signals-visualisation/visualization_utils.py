# This module contain functions needed for signal visualisation.


import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def file_paths(files_dir):
    """Generates paths to files
    Args: files_dir - files directory
    returns a list containing paths to files
    """

    paths = [os.path.join(files_dir, pulse_file).replace('\\', '/') for pulse_file in os.listdir(files_dir)]
    
    return paths


def file_validity(s, num_samples, threshold):
    """Determine validity of files
    Args: s- loaded signal
          num_samples- number of saved samples per file
          threshold- magnitude of threshold voltage
    returns 'True' if a file is valid and
    'False' if file is invalid
    """
    
    #filter files sampled and saved when there is bus conflicts in the Raspberry Pi and
    #files longer or shorter than specified number of samples
    if np.max(s) < threshold and np.min(s) > -threshold and len(s) == num_samples: 
        valid_file = True
        return valid_file
        
    else:
        valid_file = False
        if np.max(s) > threshold or np.min(s) < -threshold:
            print('File saved when there was a bus conflict. Searching for another file')

        if len(s) != num_samples:
            print('Expected {} samples. Got {} samples instead'.format(num_samples, len(s)))
            print('Confirm the number of samples in the files and the num_samples in the .ini file are the same')
            print('Searching for another file')
            
        return valid_file
    


def random_file_load(files_dir, num_samples, threshold):
    """Loads a random file
    Args: files_dir- files directory
          num_samples- number of saved samples per file
          threshold- magnitude of voltage threshold
    returns loaded signal
    """
    
    paths = file_paths(files_dir)
    
    valid_file = False
    
    while not valid_file:
        indx = np.random.randint(0, len(paths))
        file_path = paths[indx]

        s = np.array(list(map(float, pd.read_csv(file_path)['VC mem handle 5'].tolist()[1:-1])))
        
        valid_file = file_validity(s, num_samples, threshold)
        
    return s
            
            

def graph_plot(x, y, plot_title, fig_size):
    """Plots graph of x against y
    Args: x- independent variable (time)
          y- dependent variable (voltage)
          plot_title- title of the plot
          x_label- x axis label
          y_label- y axis label
          fig_size- size of figure (float, float)
    """
    
    plt.figure(figsize=fig_size)
    plt.plot(x, y)
    plt.axhline(color = 'red', linewidth=2)
    plt.xlabel('$Time\ (ms)$', fontsize=15)
    plt.ylabel('$Voltage\ (V)$', fontsize=15)
    plt.title(plot_title, fontsize=15)
    plt.show()
    
    
          


def single_plot(t, s, plot_title):
    """Plots a single plot
    Args: t- an array of sampling time interval
          s- an array of sampled values of signal
          plot_title - title of the plot
    """

    fig_size = (10,5)
    s_indx, e_indx = 0, len(s)
    graph_plot(t, s, plot_title, fig_size)


    
def edge_dup(edges, overlap_threshold):
    """Detect duplicated edges and removes them
    Args: edges- list of detected edges
          overlap_threshold- overlap threshold
    returns filtered edges
    """
    
    for edge in edges:
        for _edge in edges:
            dif = edge - _edge
            if edge != _edge and abs(dif) < overlap_threshold:
                #print('Dup detected,', edge, _edge)
                edges.remove(max([edge, _edge]))
                             
    return edges

def edge_type_verification(s, t, edges, edge_type, win_size, prior_samples):
    """Filters out mislabled edges
    Args: s- sampled signal
          t- an array of sampling time interval
          edges- list of detected edges
          edge_type- type of edge
          win_size- window size
          prior_samples- number of samples prior the edge
    """
    
    for edge in edges:
        _, win = win_generator(s, t, edge, win_size, 1000)
        if edge_type == 'rising':
            if np.max(abs(np.diff(win))) != np.max(np.diff(win)): #or np.min(win) < -1:  #check of the np.min works for all conditions
                #print('Misleading rising edge detected', edge)
                edges.remove(edge)
        
        else:
            if np.max(abs(np.diff(win))) != -np.min(np.diff(win)): #or np.min(-np.abs(win)) > -1:  #check of the np.min works for all conditions
                #print('Misleading falling edge detected', edge)
                edges.remove(edge)
    #print(edge_type, edges)           
    return edges
            
     
                
            
    
    
def edges_detection(s, t, num_edges, win_size, prior_samples, overlap_threshold):
    """Detects rising and falling edges in a signal
    Args: s- sampled signal
          t- an array of sampling time interval
          num_edges - number of edges
          win_size- window size
          prior_samples- number of samples prior the edge
          overlap_threshold- overlap threshold
    returns a dictionary containing indices of
    num_edges rising and num_edges falling edges
    """
    
    gradients = np.diff(s)
    falling_edges = np.argsort(gradients)[:num_edges].tolist()
    falling_edges = edge_dup(falling_edges, overlap_threshold)
    falling_edges = edge_type_verification(s, t, falling_edges, 'falling', win_size, prior_samples)
    rising_edges = np.argsort(gradients)[-num_edges:].tolist()
    rising_edges = edge_dup(rising_edges, overlap_threshold)
    rising_edges = edge_type_verification(s, t, rising_edges, 'rising', win_size, prior_samples)

    edges_indices = {'rising_edges':rising_edges,
             'falling_edges':falling_edges}

    return edges_indices
        


def win_generator(s, t, edge_indx, win_size, prior_samples):
    """Slices the signal and time arrays
    Args: s- sampled signal
          t- an array of sampling time interval
          edge_indx- index of an edge
          win_size- window size
          prior_samples- number of samples prior the edge
    """
    
    s_indx = edge_indx - prior_samples #start index
    e_indx = edge_indx + win_size #end index

    if s_indx < 0:
        s_indx = 0
        
    t_seg = t[s_indx: e_indx]
    v_seg = s[s_indx: e_indx]
    
    return t_seg, v_seg
    
    


def edges_plots(s, t, edges, win_size, prior_samples):
    """Plots the rising and falling edges of a signal
    Args: s- an array of sampled values of signal
          t- an array of sampling time interval
          edges- a dictinary of edges' indices
          win_size- window size
          prior_samples- number of samples prior the edge
    """
    
    fig_size = mpl.rcParams['figure.figsize']
    for edge in edges:
        if 'rising' in edge:
            plot_title = 'Rising edge'
            for indx in edges[edge]:
                t_seg, s_seg = win_generator(s, t, indx, win_size, prior_samples)
                graph_plot(t_seg, s_seg, plot_title, fig_size)

        else:
            plot_title = 'Falling edge'
            for indx in edges[edge]:
                t_seg, s_seg = win_generator(s, t, indx, win_size, prior_samples)
                graph_plot(t_seg, s_seg, plot_title, fig_size)
                

                    
            
def signal_sum(files_dir):
    """Computes the sum sampled values.
    Args: files_dir- files directory
    returns a dictionary containing sum of sampled signals
    """
    paths = file_paths(files_dir)
    sums = {}
    for file_path in paths:
        signal_sum = np.sum(np.array(list(map(float, pd.read_csv(file_path)['VC mem handle 5'].tolist()[1:-1]))))
        sums[file_path] = signal_sum
        
    return sums

     