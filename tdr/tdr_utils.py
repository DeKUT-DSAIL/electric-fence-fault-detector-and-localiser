# This module contain functions needed for time domain reflectometry.


import os
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
            #print('File saved when there was a bus conflict. Searching for another file')
            pass

        if len(s) != num_samples:
            print('Expected {} samples. Got {} samples instead'.format(num_samples, len(s)))
            print('Confirm the number of samples in the files and the num_samples in the .ini file are the same')
            print('Searching for another file')
            
        return valid_file
    
def file_loader(file_path):
    """Loads signal file
    Args: file_path- path to the file
    Returns loaded signal
    """
    
    s = np.array(list(map(float, pd.read_csv(file_path)['VC mem handle 5'].tolist()[1:-1])))
    
    return s
    


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

        s = file_loader(file_path)
        
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

def edge_validity(s, t, edges, edge_type, win_size, prior_samples):
    """Filters out mislabled edges and edges close to end of the signal
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
        
        if edge > 9900:
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
    falling_edges = edge_validity(s, t, falling_edges, 'falling', win_size, prior_samples)
    
    rising_edges = np.argsort(gradients)[-num_edges:].tolist()
    rising_edges = edge_dup(rising_edges, overlap_threshold)
    rising_edges = edge_validity(s, t, rising_edges, 'rising', win_size, prior_samples)

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
    


def single_plot(t, s, plot_title):
    """Plots a single plot
    Args: t- an array of sampling time interval
          s- an array of sampled values of signal
          plot_title - title of the plot
    """

    fig_size = (10,5)
    s_indx, e_indx = 0, len(s)
    graph_plot(t, s, plot_title, fig_size)
    
    
    
def sig_derivative(s_seg, t_seg, display=True):
    """Computes derivative of the signal using finite difference
    Args: s_seg- signal segment
          t_seg- time segment
          display- whether to plot or not
    Returns derivative of signal
    """
    
    s_derivative = np.diff(s_seg)
    
    if display:
        plt.figure(figsize=(10,5))
        plt.plot(t_seg, s_seg, 'b')
        plt.plot(t_seg[:-1], s_derivative, 'r')
        plt.xlabel('$Time (s)$', fontsize=15)
        plt.legend(['v(t)', r'$\frac{dv(t)}{dt}$'], fontsize=15)
        plt.show()
        
    return s_derivative



def edge_peaks_error(t_seg, peaks):
    """Checks if the peaks are both from a signal's rising/ falling edge
    and adjust the peaks
    Args: t_seg- time array segment
          peaks- indices of sorted signal numerical derivative
    returns peaks cleaned peaks
    """
    
    p1, p2, p3 = 0, 1, 2 #first 3 peaks indices
    
    t_delay = t_seg[peaks[p2]] - t_seg[peaks[p1]]
    
    #check if the first two peaks are both from a falling/ rising edge
    if abs(round(t_delay, 6)) == 3.2e-05 or abs(round(t_delay, 6)) == 6.5e-05:
        #print('First two peaks are Double edge peaks', t_delay)
        p3 = 1
        
        if peaks[p1] > peaks[p2]:
            peaks = np.delete(peaks, p1)
            #print('Peak shifted to rising/falling edge start point')
            
        else:
            peaks = np.delete(peaks, p2)
                  
    
    #check if the first and third peaks are both from a falling/ rising edge
    t_delay = t_seg[peaks[p3]] - t_seg[peaks[p1]]
    
    if abs(round(t_delay, 6)) == 3.2e-05 or abs(round(t_delay, 6)) == 6.5e-05:
        #print('First and third peaks are Double edge peaks', t_delay)
        
        if peaks[p1] > peaks[p3]:
            peaks = np.delete(peaks, p1)
            #print('Peak shifted to rising/falling edge start point')
            
        else:
            peaks = np.delete(peaks, p3)
        
    return peaks
    


def secondary_reflection(t_seg, peaks):
    """Checks if the second peak is from a secondary reflection and 
    shift the indices to pick the primary reflection
    Args: t_seg- time array segment
          peaks- indices of sorted signal numerical derivative
    returns rectified peaks indices
    """
    
    p1, p2, p3 = 0, 1, 2 #first 3 peaks indices
    
    l_tolerance = 1.8
    u_tolerance = 3.2
    
    t_delay = t_seg[peaks[p2]] - t_seg[peaks[p1]] #time delay between first peak and second peak
    
    t_delay_check = t_seg[peaks[p3]] - t_seg[peaks[p1]] #time delay between first peak and third peak
    
    ratio = abs(t_delay / t_delay_check)
    #print(ratio)
    
    if ratio > l_tolerance and ratio < u_tolerance:
        #print('Peak 2 is a secondary peak')        
        peaks = np.delete(peaks, p2)
        
    return peaks
    
    
    
    
def time_delay(s_seg, t_seg, display=True):
    """Detects changepoints in the signal and computes time delay
    Args: s_seg- signal segment
          t_seg- time segment
          display- whether to plot or not
    Returns the time delay
    """
    
    s_d = sig_derivative(s_seg, t_seg, display=False)
    
    peaks = np.argsort(np.abs(s_d))[::-1]
    
    peaks = edge_peaks_error(t_seg, peaks)
    
    peaks = secondary_reflection(t_seg, peaks)
    
    p1, p2 = 0, 1

    t_delay = abs(t_seg[peaks[p2]] - t_seg[peaks[p1]]) 
    
        
    if display:
        plt.figure(figsize=(10,5))
        plt.plot(t_seg, s_seg)
        plt.xlabel('$Time\ (s)$', fontsize=15)
        plt.ylabel('$Voltage\ (V)$', fontsize=15)
        plt.axvline(x = t_seg[peaks[p1]], ymin=0, ymax=1, color = 'r', label = 'axvline - full height')
        plt.axvline(x = t_seg[peaks[p2]], ymin=0, ymax=1, color = 'r', label = 'axvline - full height')
        l_limit = min(s_seg) - 0.5
        u_limit = max(s_seg) + 0.5
        plt.ylim((l_limit, u_limit))
        plt.show()
        
        
    return t_delay
    
    
def fault_point(t_delay, v_f, c=3e8):
    """Computes the point of the fault
    Args: t_delay- time delay in milliseconds
          v_f- velocity factor
          c- speed of light in vacuum in m/s
    Returns the point of fault in metres
    """
    
    fault_point = 0.5 * v_f * c * t_delay * 1e-3 #distance to the fault in metres
    
    return fault_point

def eliminate_outliers(t_delays, threshold = 6.5e-5):
    """Eliminates outlying time delays obtained from a signal
    Args: t_delays- an array of time delays
          threshold- maximum allowed distance from median
    returns time delays
    """
    #6.5e-5 >> two samples apart
    dist = np.abs(t_delays - np.median(t_delays))
    t_delays = t_delays[dist<threshold]
    
    return t_delays


def avg_delay(edges_indices, win_size, s, t, prior_samples, display=True):
    """Computes average time delay from all rising and falling
    edges of a signal
    Args: edges_indices- dictionary containing indices of edges indices of a signal
          s- sampled signal
          t- an array of sampling time interval
          edge_indx- index of an edge
          win_size- window size
          prior_samples- number of samples prior the edge
          display- whether to plot or not
    Returns average time delay and a list of time delays of each edge
    """
    
    t_delays = []
    indices = [value for values in edges_indices.values() for value in values]

    for edge_indx in indices:
        t_seg, v_seg = win_generator(s, t, edge_indx, win_size, prior_samples)
        t_delay = time_delay(v_seg, t_seg, display=display)
        t_delays.append(t_delay)
        
    t_delays  = np.array(t_delays)
        
    t_delays = eliminate_outliers(t_delays)

    avg_t_delay = np.mean(t_delays)
    
    return avg_t_delay, t_delays