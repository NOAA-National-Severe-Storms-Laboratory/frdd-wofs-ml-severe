from glob import glob
import xarray as xr
import numpy as np
from os.path import join
import os
from WoF_post.wofs.plotting.wofs_colors import WoFSColors
from WoF_post.wofs.plotting.wofs_levels import WoFSLevels
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from skimage.measure import regionprops
from scipy.ndimage import maximum_filter, gaussian_filter, minimum_filter

import sys
sys.path.insert(0, '/home/monte.flora/python_packages/MontePython')
import monte_python


MRMS_PATH = '/work/brian.matilla/WOFS_2021/MRMS/RAD_AZS_MSH_AGG/'
WOFS_PATH = '/work/mflora/SummaryFiles/'

def get_files(date, init_time, t):
    WOFS_OFFSET = 6 
    MRMS_OFFSET = 18 

    WOFS_PATH = '/work/mflora/SummaryFiles/'
    MRMS_PATH = '/work/brian.matilla/WOFS_2021/MRMS/RAD_AZS_MSH_AGG/'
    
    wofs_t = t + WOFS_OFFSET
    mrms_t = t + MRMS_OFFSET
    mrms_begin_t =mrms_t - 6 
    mrms_files = [glob(join(MRMS_PATH, date, init_time, f'wofs_RAD_{tt:02d}*'))[0] for tt in 
                  range(mrms_begin_t, mrms_t+1)]
                  
    mrms_dbz = np.max([xr.load_dataset(f)['dz_cress'].values for f in mrms_files], axis=0)
    wofs_file = glob(join(WOFS_PATH, date, init_time, f'wofs_ENSEMBLETRACKS_{wofs_t:02d}*'))[0]

    ds = xr.load_dataset(wofs_file)
    ensemble_tracks = ds['w_up__ensemble_tracks'].values
    probs = ds['w_up__ensemble_probabilities'].values
        
    return mrms_dbz, ensemble_tracks, probs    



def identify_mrms_tracks(mrms_dbz, min_thresh=44):

    storm_labels, object_props = monte_python.label( input_data = mrms_dbz,
                                   method ='watershed', 
                                   return_object_properties=True, 
                                   params = {'min_thresh': min_thresh,
                                             'max_thresh': 75,
                                             'data_increment': 5,
                                              'area_threshold': 1500,
                                            'dist_btw_objects': 25 } )
    # Quality Control 
    qcer = monte_python.QualityControler()
    qc_params = [('min_area', 18), ('max_thresh', [45, 99])]
    qc_labels, qc_objects_props = qcer.quality_control(mrms_dbz, storm_labels, object_props, qc_params)
    
    return qc_labels, qc_objects_props


def get_all_labels(arr):
    """Determine the unique labels in a 2D labelled array."""
    if np.max(arr) > 0:
        all_labels = np.unique(arr)[1:]
    else:
        all_labels = []
        
    return all_labels

def add_contingency_table_compos(fcst_labels, obs_labels, 
                                 matched_fcst, matched_obs):
    
    table = np.zeros((2,2))
    # Hits, False Alarms
    # Misses, CNs 
    
    all_fcst_labels = get_all_labels(fcst_labels)
    all_obs_labels = get_all_labels(obs_labels)
        
    # Hits (matched fcst labels)
    table[0,0] = len(matched_fcst)
        
    # False Alarms (fcst labels unmatched)
    table[0,1] = (len(all_fcst_labels) - len(matched_fcst))
        
    # Misses (obs unmatched)
    table[1,0] = (len(all_obs_labels) - len(matched_obs))
    
    return table


def new_id(input_data, remove_low=True):
    param_set = [ {'min_thresh': 0,
                   'max_thresh': 100,
                   'data_increment': 10,
                   'delta': 0,
                   'area_threshold': 1000,
                   'dist_btw_objects': 125 },
             
             {'min_thresh': 20,
                   'max_thresh': 100,
                   'data_increment': 5,
                   'delta': 0,
                   'area_threshold': 500,
                   'dist_btw_objects': 15 },

            ]

    params = {'params': param_set }

    # Less than 2/18 = 0.11
    new_input_data = np.copy(input_data)
    if remove_low:
        new_input_data[input_data<=0.12] = 0

    new_input_data = maximum_filter(new_input_data, size=2)
    new_input_data = gaussian_filter(new_input_data, 1)*100

    storm_labels, new_object_props = monte_python.label(  input_data = new_input_data, 
                       method ='iterative_watershed', 
                       return_object_properties=True, 
                       params = params,  
                       )
    
    # Reduce the object size due to the maximum filter and gaussian filter 
    storm_labels = minimum_filter(storm_labels, size=3)
    new_object_props = regionprops(storm_labels, storm_labels)
    
    return storm_labels, new_input_data, new_object_props


class WoFSVerifier:
    def __init__(self):
        
        self.hits = 0
        self.false_alarms = 0
        self.misses = 0
        
    def get_scores(self):
        return self.pod(), self.far(), self.csi()
    
    def get_all_labels(self, arr):
        """Determine the unique labels in a 2D labelled array."""
        if np.max(arr) > 0:
            all_labels = np.unique(arr)[1:]
        else:
            all_labels = []
        
        return all_labels
        
    def add_contingency_table_compos(self, fcst_labels, obs_labels, matched_fcst, matched_obs):
        all_fcst_labels = self.get_all_labels(fcst_labels)
        all_obs_labels = self.get_all_labels(obs_labels)
        
        # Hits (matched fcst labels)
        self.hits += len(matched_fcst)
        
        # False Alarms (fcst labels unmatched)
        unique_fcst = np.unique(matched_fcst)
        self.false_alarms += (len(all_fcst_labels) - len(unique_fcst))
        
        # Misses (obs unmatched)
        self.misses += (len(all_obs_labels) - len(matched_obs))
    
    def pod(self):
        "Probability of Detection"
        return self.hits / (self.hits + self.misses)
    
    def far(self):
        "False Alarm Ratio"
        return self.false_alarms / (self.hits + self.false_alarms)
    
    def csi(self):
        "Critical Success Index"
        return self.hits / (self.hits + self.misses + self.false_alarms)