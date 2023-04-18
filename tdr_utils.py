import numpy as np
import pandas as pd
from collections import namedtuple


def file_validity(s, num_samples, v_peak):
    """Determine validity of files
    Args: s- loaded signal
          num_samples- number of saved samples per file
          v_peak- peak voltage
    returns 'True' if a file is valid and
    'False' if file is invalid
    """
    
    #filter files sampled and saved when there is bus conflicts in the Raspberry Pi and
    #files longer or shorter than specified number of samples
    if np.max(s) < v_peak and np.min(s) > -v_peak and len(s) == num_samples: 
        valid_file = True
        return valid_file
        
    else:
        valid_file = False            
        return valid_file
    
    
def file_loader(file_path):
    """Loads signal file
    Args: file_path- path to the file
    Returns loaded signal
    """
    
    s = pd.read_csv(file_path, header=None).values.flatten()
    
    return s


def random_file_load(paths, num_samples, v_peak):
    """Loads a random file
    Args: paths- files' paths
          num_samples- number of saved samples per file
          v_peak- peak voltage
    returns loaded signal
    """
    
    valid_file = False
    
    while not valid_file:
        indx = np.random.randint(0, len(paths))
        file_path = paths[indx]

        s = file_loader(file_path)
        
        valid_file = file_validity(s, num_samples, v_peak)
    return file_path, s



def edge_detection(s, st_dev, pulse_freq, duty_cycle, sampling_rate, rising_edge_v, num_samples, win_size, negative_threshold):
    """Extract indices of the rising edges of a signal
    Args: s- signal
          st_dev- standard deviation of the tdr signal
          pulse_freq- frequency of the tdr signal
          duty_cycle- duty cycle of the tdr signal
          sampling_rate- sampling rate
          rising_edge_v- standard deviation weighting factor
          num_samples- number of samples
          win_size- window size
          negative_threshold- threshold of negatively reflected signal
    returns a binary array of ones and zeros (edges=1) and indices of the 
    rising edges
    """
    
    if np.min(s) < negative_threshold:
        fault_type = 'Short circuit'
        
    else:
        fault_type = 'Open circuit'
    
    sig_diff = np.abs(np.diff(s)) - st_dev * rising_edge_v
    
    edges = np.where(sig_diff<0, 0, 1)
    
    e_indices = np.where(edges>0)[-1]
    
    e_diff = np.diff(e_indices)
    
    on_win = int(duty_cycle * (1 / pulse_freq) * sampling_rate) 
    
    e_indices = np.delete(e_indices, np.where(e_diff<on_win)[-1]+1) #eliminate duplicate edges
    e_indices = np.delete(e_indices, np.where(e_indices > num_samples-win_size)) #eliminate edges so close to the end of the signal  
    return edges, e_indices, fault_type




def changepoint(s, c_len, e_indices, sampling_rate, c, a_vf, win_size, reflection_edge_threshold):
    """Dedtect changepoints due to the reflected signal
    Args: s- signal
          c_len- length of the cable
          e_indices- indices of the rising edges
          sampling_rate- sampling rate
          c- speed of light in vacuum
          a_vf- approximate velocity factor
          win_size- window size
          reflection_edge_threshold- threshold for distance apart between change points 
    returns indices of change points and a signal's segments of interest
    """

    cp_indices = []
    segs_interest = []
    
    for indx in e_indices:
        s_indx = indx + 2
        e_indx = indx + win_size
        
        seg_diff = np.abs(np.diff(s[s_indx:e_indx])) - reflection_edge_threshold
        
        edges = np.where(seg_diff<0, 0, 1)
        
        e_indices = np.argmax(edges>0)
        
        cp_indx = indx + e_indices + 2
        
        if cp_indx - indx > 2:
            cp_indices.append(cp_indx)
            seg_interest = [indx, cp_indx]
            segs_interest.append(seg_interest)
            
    if len(cp_indices) == 0:
        raise Exception('No change point detected!')
        
        
    return cp_indices, segs_interest
        
        
        
def time_delay(t, segs_interest):
    """Compute time delay of reflected signal
    Args: t- time
          segs_interst- segments of interest
    returns an array of time delays from the rising edges and
    an average of the delays.
    """
    
    t_delays = []
    for seg in segs_interest:
        t_delay = t[seg[1]] - t[seg[0]]
        t_delays.append(t_delay)   
    if len(t_delays) == 0:
            raise Exception('No time delay obtained!')
        
    for t in t_delays:
        if t / np.min(t_delays) > 1.8:
            t_delays.remove(t)
            
    avg_t_delay = sum(t_delays) / len(t_delays)
        
    return t_delays, avg_t_delay


def velocity_factor(D, r, c):
    """Computes the velocity factor of a
    cable
    Args: D- distance of separation in m
          r- radius of the cables in m
          c- velocity of light in vacuum
    """
    
    C = (np.pi * 8.854e-12) / (np.log(D/r)) #Capacitance per unit length
    
    L = 4e-7 * np.log(D / (r * np.exp(-.25))) #Inductance per unit length
    
    vf = 1 / (c * (L * C) ** .5)
    
    return vf



def fault_point(t_delay, c, D, r):
    """Computes the point of the fault
    Args: t_delay- time delay in milliseconds
          c- speed of light in vacuum in m/s
          D- distance of separation in m
          r- radius of the cables in m
    Returns the point of fault in metres
    """
    
    vf = velocity_factor(D, r, c)
    
    fault_point = 0.5 * vf * c * t_delay * 1e-3 #distance to the fault in metres
    
    return fault_point



def performance_metrics(e_values, p_values, metrics=['mse', 'mae', 'irm'], margin=5):
    """Computes the performance of the tdr algorithm using the specified
    algorithms. The acceptable metrics are mean squared error (mse), mean 
    absolute error (mae) and inner ratio metrics (irm)
    Args: e_values- list/array of expected values
          p_values- list/array of predicted values
          metrics- a list containing metrics
          margin- error margin for inner ratio metrics
    Returns the mean squared error
    """
    
    if str(type(e_values))[8:-2] != 'numpy.ndarray' and type(e_values) != list and type(e_values) != int and type(e_values) != float  and str(type(e_values))[8:-4] != 'numpy.float' and str(type(e_values))[8:-4] != 'numpy.int':
        raise Exception("e_values should be an integer, a list or an array. {} was provided instead". format(str(type(e_values))[8:-2].capitalize()))
            
    if str(type(p_values))[8:-2] != 'numpy.ndarray' and type(p_values) != list and type(p_values) != int and type(p_values) != float and str(type(p_values))[8:-4] != 'numpy.float' and str(type(p_values))[8:-4] != 'numpy.int':
        raise Exception("p_values should be an integer, a list or an array. {} was provided instead". format(str(type(p_values))[8:-2].capitalize()))
        
    if type(e_values) == int or type(e_values) == float or str(type(e_values))[8:-4] == 'numpy.float' or str(type(e_values))[8:-4] == 'numpy.int':
        if type(p_values) != int and type(p_values) != float and str(type(p_values))[8:-4] != 'numpy.float' and str(type(p_values))[8:-4] != 'numpy.int':
            if len(p_values) != 1:
                raise Exception("e_values is an integer. Expected a list/array with a shape of (1,) for p_values.  Got a {} {} instead.". format(np.shape(p_values), str(type(p_values))[8:-2]))
     
    if  type(p_values) == int or type(p_values) == float or str(type(p_values))[8:-4] == 'numpy.float' or str(type(p_values))[8:-4] == 'numpy.int':
        if type(e_values) != int and type(e_values) != float and str(type(e_values))[8:-4] != 'numpy.float' and str(type(e_values))[8:-4] != 'numpy.int':
            if len(e_values) != 1:
                raise Exception("p_values is an integer. Expected a list/array with a shape of (1,) for e_values.  Got a {} {} instead.". format(np.shape(e_values), str(type(e_values))[8:-2]))
    
    if type(e_values) == int or type(e_values) == float or str(type(e_values))[8:-4] == 'numpy.float' or str(type(e_values))[8:-4] == 'numpy.int':
        e_values = np.array([e_values])
 
    if  type(p_values) == int or type(p_values) == float or str(type(p_values))[8:-4] == 'numpy.float' or str(type(p_values))[8:-4] == 'numpy.int':
        p_values = np.array([p_values])

    if type(e_values) == list:
        e_values = np.array(e_values)
            
    if str(type(p_values))[8:-2] == list:   
        p_values = np.array(p_values)
     
    #elements_check(e_values, p_values)
    metrics_values_list = []
    if len(e_values) == len(p_values):
        metrics_dic = {}
        
        for metric in metrics:
            if metric.lower() != 'mae' and metric.lower() != 'mse' and metric.lower() != 'irm':
                raise ValueError("{} is not a recognised value for metrics. The recognised values are 'mse' for mean square error, 'mae' for mean absolute error an 'irm' for inner ratio metric".format(metric))
            
            name = metric.upper()
            
            if metric.lower() == 'mse':            
                mse = np.mean((e_values - p_values) ** 2)
                rmse = mse ** .5
                metrics_dic['mse'] = mse
                metrics_dic['rmse'] = rmse
                
            elif metric.lower() == 'mae':
                mae = np.mean(np.abs((e_values - p_values)))
                metrics_dic['mae'] = mae

                
            elif metric.lower() == 'irm':
                divs_list = []
                divs = np.abs(p_values - e_values)
                for div in divs:       
                    if div <= margin:
                        divs_list.append(div)

                irm = len(divs_list) / e_values.size
                metrics_dic['irm'] = irm
                
            Metrics = namedtuple('Metrics', metrics_dic)
            
            performance = Metrics(**metrics_dic)
        
    else:
        raise Exception("Expected lists/arrays of equal length. Lists/arrays of lengths {} and {} were provided.".format(len(e_values), len(p_values)))
        
    return performance