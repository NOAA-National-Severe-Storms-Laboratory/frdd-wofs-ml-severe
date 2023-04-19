#======================================================
# Extracts ML features using storm tracks. 
# 
# Author: Montgomery Flora (Git username : monte-flora)
# Email : monte.flora@noaa.gov 
#======================================================

# Python Modules
import itertools 
import warnings

# Third Party Modules
import pandas as pd 
from skimage.measure import regionprops_table
from scipy.ndimage import maximum_filter
import numpy as np
import xarray as xr 

# Temporary until the ML models are re-trained 
# with the new naming convention!!!
_base_module_path = '/home/monte.flora/python_packages/WoF_post'
import sys
sys.path.insert(0, _base_module_path)
from wofs.post.utils import convert_to_seconds


class StormBasedFeatureExtracter():
    """
    StormBasedFeatureExtracter is designed to extract storm and environmental data 
    from a storm object using the method outlined in Flora et al. (2021). 
    
    Attribute
    ----------------
    ml_config : dict 
        A dictionary containing the following: 
            1. A list of min_vars; the variables to be computed as a time-min composite 
                or as ensemble minimum. 
            2. Basline thresholds and NMEP sizes 
            3. morphological variables to extract from the skimage.measure.regionprops object 
    """
    def __init__(self, ml_config, cond_var='w_up__time_max', cond_var_thresh=12.0, dx = 3):
        self.ml_config = ml_config
        self.dx = dx 
        self.cond_var = cond_var
        self.cond_var_thresh = 12.0 
        
        self.spatial_percentiles_for_amps = [10,90]
        
    def extract(self, storm_objects, intensity_img, storm_data, env_data, init_time, updraft_tracks=None): 
        """
        Using labeled data, extract spatial- and amplitude-based features from ensemble data. 
        Environmental data is only extract as spatial-based features while the intra-storm
        data is extracted as spatial and amplitude based features. 
        
        Parameters
        --------------
        storm_objects : array of shape (NY, NX)
            Labeled storm tracks.
        
        intensity_img : array of shape (NY, NX)
            Input array for the labeled storm tracks. 
            
        storm_data : dict of {var : array of shape (NT, NE, NY, NX)}
            The ensemble intra-storm data at multiple times.
        
        env_data : dict of {var : array of shape (NE, NY, NX)}
            The ensemble environmental data at the beginning of
            the forecast period.
        
        init_time : str of format 'HHmm'
            The initialization time for the forecast. 
        
        Returns
        --------------
        dataframe : pandas.DataFrame
            The ML dataset where the rows are for each object 
            and columns are the various ML features. 
        """
        if not self._is_there_one_object(storm_objects):
            return None 
    
        # Compute the object properties
        object_props_df = self.get_object_properties(storm_objects, intensity_img)
        labels = object_props_df['label']
        if updraft_tracks is not None:
            results = self.area_ratio(storm_objects, updraft_tracks, labels)
            object_props_df['area_ratio'] = results[0]
            object_props_df['avg_updraft_track_area'] = results[1]
            
        # Get the time-composite intra-storm data 
        storm_data_time_composite = self._compute_time_composite(storm_data)
        
        # Compute the baseline NMEP 
        baseline_probs = self._get_baseline(storm_data_time_composite)
        
        # Combine the time composites with the environmental data 
        combined_data = {**env_data, **storm_data_time_composite}
        
        # Compute ens. statistics (data is still 2d at this point). 
        ensemble_data = self._compute_ens_stats(combined_data)
        
        # Extract the spatial-based features.
        df_spatial = self.extract_spatial_features_from_object( 
                ensemble_data, 
                storm_objects, 
                labels
        )
    
        # Extract the amplitude-based features
        df_amp = self.extract_amplitude_features_from_object( 
                storm_data_time_composite, 
                storm_objects, 
                labels, 
                cond_var=self.cond_var, 
                cond_var_thresh=self.cond_var_thresh
        )
        
        # Extract the baseline predictions
        df_nmep = self.extract_spatial_features_from_object( 
            baseline_probs, 
            storm_objects, labels, 
            stat_funcs=[(np.nanmax,None)], 
            stat_func_names = ['__prob_max'] )
        
        # Concatenate everything into a single dataframe.
        dataframe = pd.concat([df_spatial, df_amp, object_props_df, df_nmep], axis=1)

        # Retuns the initialization time as string of format (HHmm)
        dataframe['Initialization Time'] = [str(init_time)]*np.shape(dataframe)[0]
 
        # Reset the indices for feathering.
        dataframe.reset_index(drop=True, inplace=True)

        return dataframe

    def to_dataframe(self, arr, columns):
        """Returns arr as a pandas.DataFrame"""
        return pd.DataFrame(arr, columns=columns)

    def _is_there_one_object(self, storm_objects):
        """Check if there is at least one storm object"""
        return np.max(storm_objects) > 0 
        
    def _compute_ens_stats(self, data):
        """Compute the ensemble mean, standard dev., and ensemble maximum/minimum"""
        data_vars = list(data.keys())
        
        # Compute the ensemble mean 
        with warnings.catch_warnings():
            # Supress the Runtime Warnings.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            ens_mean_data = {
                var+'__ens_mean': np.nanmean(data[var], axis=0)
                for var in data_vars
            }

            # Compute the ensemble standard deviation 
            ens_std_data = {
                var+'__ens_std': np.nanstd(data[var], axis=0, ddof=1)
                for var in data_vars
            }
        
            # Compute the ensemble maximum 
            ens_max_data = {
                var+'__ens_max': np.nanmax(data[var], axis=0)
                for var in data_vars if 'max' in var
            }
        
            # Compute the ensemble minimum for min variables 
            ens_min_data = {
            var+'__ens_min': np.nanmin(data[var], axis=0)
            for var in data_vars if 'min' in var
            }
        
        # Combine the ensemble statistics into a single dict. 
        ensemble_data = {**ens_mean_data, **ens_std_data, **ens_max_data, **ens_min_data}
        
        return ensemble_data
        
    def _compute_time_composite(self, storm_data):
        """Compute the time-maximum (minimum) variables"""
        time_max_strm_data = {
            var+'__time_max': np.nanmax(storm_data[var], axis=0)
            for var in storm_data.data_vars if var not in self.ml_config['ENS_MIN_VARS']
        }
        
        # Compute the time-minimum variables 
        time_min_strm_data = {
        var+'__time_min': np.nanmin(storm_data[var], axis=0)
        for var in self.ml_config['ENS_MIN_VARS']
        }
        
        # Combine back the time-max and -min data
        storm_data_time_composite = {**time_max_strm_data, **time_min_strm_data} 
        
        return storm_data_time_composite
        
    
    def _get_baseline(self, data):
        """Compute the baseline uncalibrated NMEP probabilities"""
        UH_THRESHS = self.ml_config['UH_THRESHS']
        WIND_THRESHS = self.ml_config['WIND_THRESHS']
        HAIL_THRESHS = self.ml_config['HAIL_THRESHS']
        
        NMEP_SIZES = self.ml_config['NMEP_SIZES']
        
        uh_probs = {f'uh_nmep_>{t}_{n*self.dx}km': self.calc_ensemble_probs(data['uh_2to5_instant__time_max'], 
                                                                            thresh=t, size=n)
                    for t,n in itertools.product(UH_THRESHS, NMEP_SIZES)} 
        
        wnd_probs = {f'wind_nmep_>{t}_{n*self.dx}km': self.calc_ensemble_probs(data['ws_80__time_max'], 
                                                                               thresh=t, size=n)
                    for t,n in itertools.product(WIND_THRESHS, NMEP_SIZES)} 
        
        hail_probs = {f'hail_nmep_>{t}_{n*self.dx}km': self.calc_ensemble_probs(data['hailcast__time_max'], 
                                                                                thresh=t, size=n)
                    for t,n in itertools.product(HAIL_THRESHS, NMEP_SIZES)} 
        
        baseline_probs = {**uh_probs, **wnd_probs, **hail_probs}
    
        return baseline_probs 
    
    def apply_max_filter(self, data, size):
        """Applies maximum_filter to each ensemble member."""
        data_maxed = np.array(
            [maximum_filter(data[i,:,:], size=size) for i in range(data.shape[0])]
        )
        return data_maxed
    
    def calc_ensemble_probs(self, data, thresh, size=0):
        """
        Compute the Neighborhood Maximum Ensemble Probability (Sobash and Schwartz 2017)
        
        Parameters
        -------------
        data : array-like of shape (NE, NY, NX)
            Input ensemble data 
            
        thresh : float 
            Threshold 
        
        size : int 
            Maximum filter size 
        
        Return 
        --------------
        ensemble_probabilities : array-like (NY, NX)
            Neighborhood Maximum Ensemble Probabilities 
        """
        # Apply a max value filter to each ensemble member separately.
        data_maxed = self.apply_max_filter(data, size)
        # Binarnize the data. 
        binary_data = np.where(np.round(data_maxed, 5) > thresh,True,False)
        # Compute the ensemble probability. 
        ensemble_probabilities = np.mean(binary_data, axis=0)

        return ensemble_probabilities
        
    def extract_spatial_features_from_object( self, input_data, 
                                             labeled_img, 
                                             labels,
                                             stat_funcs=None,
                                             stat_func_names=None
                                    ):
        """ Extracts spatial statistic from within
        a labeled region for each variable. 

        Parameters
        -----------
            input_data : dict of {var : array of shape (NY, NX)}
                The input data. 
            
            labeled_img : array of shape (NY, NX)
                Labeled input image.
                
            labels : list of int
                The set of labels within labeled_img.
            
            stat_funcs : list of callables (default is None)
                The functions used to compute the spatial statistics. 
                Form = [(np.nanstd, None), (np.max), ...]
                If None, by default only the spatial average
                value is computed. 
            
            stat_func_names : list of str (default is None)
                Names to be given to the statistics. 
                E.g., ['__ens_std', '__ens_max', ...]
                If None, by default, the name for the spatial average
                is used. 
            
        Returns
        ----------
            df : pandas.DataFrame of shape ( n_objects, n_var*n_stats )
                Spatial statistics for each region. 
        """
        if stat_funcs is None:
            stat_funcs, stat_func_names = self._set_of_stat_functions(only_mean=True)
        
        n_labels = len(labels)
        var_list = list( input_data.keys() )
        n_features = len(var_list) * len(stat_funcs)
        num_of_spatial_stats = len(stat_funcs)
      
        features = np.zeros(( n_labels, len(var_list)*len(stat_funcs) ))
        
        if len(np.unique(labeled_img)) == 1 or n_labels == 0:
            return features
        else:
            feature_names = []
            
            storm_points = {label: np.where( labeled_img == label ) for label in labels}
            for i, object_label in enumerate( labels):
                n=0
                inds = storm_points[object_label]
                for var in var_list:
                    data_subset = input_data[var][inds]
                    for (func, func_param), func_name in zip(stat_funcs, stat_func_names):
                        if i == 0:
                            feature_names.append(f'{var}{func_name}')
                        features[i, n] = self._generic_function( func, 
                                                   input_data=data_subset,
                                                   parameter=func_param
                                                  )
                        n+=1
                    
        df = self.to_dataframe(features, feature_names)           
                    
        return df


    def extract_amplitude_features_from_object( self,
                                                input_data, 
                                                labeled_img,
                                                labels,
                                                cond_var=None,
                                                cond_var_thresh=None
                                                ):
        """
        Extract ensemble amplitude statistics from object. 
        Spatial maximum is computed from each ensemble members 
        within a given region. From those ensemble values, we 
        can compute various statistics (e.g., std, mean, max, etc). 
        
        Parameters
        --------------
        input_data : array of shape ()
            Ensemble data to compute the amplitude features from. 
        
        labeled_img : array of shape (NY, NX)
            Labeled data representing the ensemble storm tracks.
            
        labels : list of int
            Track labels where features are computed for. 
            Bad labels consist of those features too close to the domain boundary.
        
        Returns
        --------------
        df : pandas.DataFrame of shape (n_objects, n_var * n_stats)
            Amplitude features for each track.
        """
        func_set   = [ (np.nanstd, 'ddof') , (np.nanmean,None), ('minmax', None) ]
        func_names = [ '__amp_ens_std', '__amp_ens_mean', '__amp_ens_max']
        var_list = list(input_data.keys())
        
        n_labels = len(labels)
        n_vars = len(var_list)
        n_stats = len(func_set)
        n_features = n_vars * n_stats

        if cond_var is None:
            tags = [''] 
            features = np.zeros((n_labels, n_features),np.float32)
        else:
            # Re-position cond_var so it is the first variable
            # It is used to compute the conditional ensemble amplitude stats.
            var_list.remove(cond_var)
            var_list.insert(0, cond_var)
            tags = ['', '__cond']
            # Multiplied by 2 to account for the additional conditional variables.
            # Minus one from statistics as we are not computing the 
            # conditional maximum as it is redundant. 
            n_features = (n_vars * (n_stats)) + (n_vars * (n_stats-1))
            features = np.zeros((n_labels, n_features ), dtype=np.float32)
        
        if len(np.unique(labeled_img)) == 1 or n_labels == 0:
            return features
        else:
            feature_names = []
            storm_points = {label: np.where(labeled_img == label) for label in labels}
            for i, object_label in enumerate(labels):
                n=0
                c=0
                for var in var_list:
                    min_var = True if 'min' in var else False
                    k = 0 if min_var else 1 
                    data = input_data[var]
                    amplitudes = self.ensemble_amplitudes( data = data, 
                                                          storm_points=storm_points, 
                                                          object_label=object_label, 
                                                          min_var=min_var)
                    
                    if c == 0 and cond_var is not None:
                        # The first pass through will be the conditional variable. 
                        cond_var_amps = amplitudes
                        c+=1
                    
                    if cond_var is not None:
                        # Get the indices where the w_up time-max
                        # is > 12 m/s. This should be a good proxy
                        # for the non-missing ensemble members
                        idx = np.where(np.round(cond_var_amps,6)>=cond_var_thresh)[0]
                        indices_set = [np.arange(len(amplitudes)), idx]
                    else:
                        indices_set = [np.arange(len(amplitudes))]
                    
                    # Cycle through marginal and conditional
                    for (func, func_param), name_tag in zip(func_set, func_names):
                        _func = 'minmax' if func == 'minmax' else None
                        if func == 'minmax':
                            func = np.nanmin if min_var else np.nanmax
                            k = 0 if min_var else 1 
                            
                        for tag, inds in zip(tags, indices_set):
                            if _func == 'minmax' and tag == '__cond':
                                # There is no conditional maximum/minimum so skip this 
                                # to remove redundant features. 
                                continue
                                
                            if i == 0:
                                feature_names.append(
                                    f'{var}{name_tag}{tag}_spatial_perc_{self.spatial_percentiles_for_amps[k]}')
   
                            features[i, n] = self._generic_function( func,
                                                   input_data=amplitudes[inds],
                                                   parameter=func_param,
                                                  )
                            n+=1

        df = self.to_dataframe(features, feature_names) 
        
        return df
 
        
    def ensemble_amplitudes(self, data, storm_points, object_label, min_var=False):
        """
        Computes the spatial maximum (minimum) percentile from each ensemble member 
        within the ensemble storm track for each variable. 
        
        Parameters
        ------------
            data : array of shape (NE, NY, NX)
            storm_points : 
            object_label :  
            min_var : True/False
        
        Returns
        -------------
        amplitudes : array of shape (NE,)
        
        """
        y_set = storm_points[object_label][0]
        x_set = storm_points[object_label][1]
        percentile = self.spatial_percentiles_for_amps[0] if min_var else self.spatial_percentiles_for_amps[1]  
        amplitudes = np.nanpercentile(data[:,y_set,x_set], percentile, axis=1)  

        return amplitudes

    def _set_of_stat_functions( self, names=True, only_mean=False ):
        """ Function returns a list of function objects """
        func_set   = [  (np.nanmean,None), (np.nanpercentile, 10), (np.nanpercentile, 90) ]
        func_names = [  '__spatial_mean', '__spatial_10th',  '__spatial_90th' ]      
    
        if only_mean:
            if names:
                return [(np.nanmean, None)], [ '__spatial_mean']
            else:
                return [(np.nanmean, None)]
        else:
            if names:
                return func_set, func_names
            else:
                return func_set

    def _generic_function( self, func , input_data, parameter=None ):
        """ A function meant to implement the various function objects in 'set_of_stat_functions'"""
        with warnings.catch_warnings():
            # Supress the Runtime Warnings.
            warnings.simplefilter("ignore", category=RuntimeWarning)
        
            if parameter == 'ddof':
                if len(input_data) ==1:
                    # This accounts for conditional std. where input_data
                    # is a single value, which returns a nan. 
                    return 0.0
                else:
                    return func(input_data, ddof=1)
            
            elif parameter is not None:
                return func( input_data, parameter)
            else:
                return func(input_data)

    def get_object_properties(self, label_img, intensity_img):
        """ Returns the object properties as a pandas.DataFrame """
        properties =  self.ml_config['MORPHOLOGICAL_FEATURES'] + ['centroid', 'label'] 
        
        data = regionprops_table(label_img, intensity_img, properties) 
        df = pd.DataFrame(data)
        df = df.rename({'centroid-0' : 'obj_centroid_y', 
                'centroid-1' : 'obj_centroid_x', 
                'intensity_max' : 'ens_track_prob',
               },
               axis=1
              )
        
        return df

    
    def average_updraft_track_area(self, updraft_tracks, points):
        """Computes the average area of the updraft tracks from the individual 
        ensemble members"""
        return np.mean([np.count_nonzero(updraft_tracks[n,:,:][points])
                        for n in range(updraft_tracks.shape[0])])
    
    def area_ratio(self, ensemble_tracks, updraft_tracks, labels):
        """
        Computes the ratio of the ensemble avg. area of the individual storm tracks
        to the area of the ensemble storm track. 
    
        Parameters
        ----------------
        ensemble_tracks : array of shape (NY, NX)
        updraft_tracks : array of shape (NE, NY, NX)
    
        Returns
        ---------------
        ratios : array-like shape (n_labels)
        """
        # Initialize the ratios array 
        ratios = np.zeros(len(labels), dtype=np.float32)
        areas = np.zeros(len(labels), dtype=np.float32)
        
        # Iterate on the labels 
        for i,label in enumerate(labels):
            points = np.where(ensemble_tracks==label)
            ensemble_area = np.count_nonzero(np.where(ensemble_tracks==label,1,0))

            # Should this ensemble avg area or conditional avg? 
            avg_area = self.average_updraft_track_area(updraft_tracks, points)

            ratios[i] = avg_area / ensemble_area
            areas[i] = avg_area

        return ratios, areas
