import numpy as np
import itertools
import sys
sys.path.append( '/home/monte.flora/NEWSeProbs/misc_python_scripts')
from scipy.spatial import cKDTree


class StormBasedFeatureEngineering:
    '''
    StormBasedFeatureEngineering handles extraction for both patches and storm-based 
    '''
    def __init__( self, grid_size = 20, patches=False, ROI_STORM = 7, ROI_ENV = 15 ): 
        self.ROI_STORM = ROI_STORM   
        self.ROI_ENV = ROI_ENV 
        if patches:
            self.grid_size = grid_size
            self.delta = int(grid_size/2)
            self.dist_from_edge = 6
            self.BUFFER = self.delta + self.dist_from_edge
        else:
            self.BUFFER = ROI_STORM + 1

    def extract_spatial_features_from_object( self, input_data, 
                                             forecast_objects, 
                                             good_object_labels,
                                             mem_idx=None,
                                             only_mean=False, 
                                             stat_funcs=None,
                                             stat_func_names=None
                                    ):
        ''' Extract intra-storm state features for machine learning 
        
        ------------------
        Args:
            input_data, 
        Returns:
            features, shape = ( n_objects, n_var*n_stats )
        '''
        if stat_funcs is None:
            stat_funcs, stat_func_names = self._set_of_stat_functions(only_mean=only_mean)
        num_of_objects = len(good_object_labels)
        var_list = list( input_data.keys() )
        num_of_features = len(var_list) * len(stat_funcs)
        num_of_spatial_stats = len(stat_funcs)
      
        #n_object, n_var*n_stats
        feature_names = [ ]
        for n, atuple in enumerate( list(itertools.product(var_list, range(num_of_spatial_stats)))):
            feature_names.append(atuple[0]+stat_func_names[atuple[1]]) 

        features = np.zeros(( num_of_objects, len(var_list)*len(stat_funcs) ))
        if len(np.unique(forecast_objects)) == 1 or num_of_objects == 0:
            return features
        else:
            storm_points = {label: np.where( forecast_objects == label ) for label in good_object_labels}
            for i, object_label in enumerate( good_object_labels): 
                for n, atuple in enumerate(list( itertools.product(var_list, range(num_of_spatial_stats)))):    
                    var = atuple[0]; k = atuple[1]
                    if mem_idx is not None:
                        data = input_data[var][mem_idx,:,:]
                    else:
                        data = input_data[var]

                    ###print(var, data.shape) 
                    features[i, n] = self._generic_function( stat_funcs[k][0], 
                                                   input_data=data[storm_points[object_label]],
                                                   parameter=stat_funcs[k][1]
                                                  )
                    
        return features, feature_names #dim = (n_objects, n_var*n_stats) 



    def extract_amplitude_features_from_object( self,
                                                input_data, 
                                                forecast_objects,
                                                good_object_labels,
                                                ):
        ''' Extract ensemble amplitude statistics from object '''
        func_set   = [ (np.nanstd, 'ddof') , (np.nanmean,None)]
        func_names = [ '__ens_std_of_90th', '__ens_mean_of_90th', ]
        num_of_objects = len(good_object_labels)
        var_list = list( input_data.keys() )
        num_of_features = len(var_list) * len(func_set)
        features = np.zeros(( num_of_objects, len(var_list)*len(func_set) ))

        feature_names = [ ]
        for n, atuple in enumerate( list(itertools.product(var_list, range(len(func_set))))):
            name_tag = func_names[atuple[1]]
            if 'min' in atuple[0]:
                name_tag = name_tag.replace('9', '1')
            feature_names.append(atuple[0]+name_tag) 

        if len(np.unique(forecast_objects)) == 1 or num_of_objects == 0:
            return features
        else:
            storm_points = {label: np.where( forecast_objects == label ) for label in good_object_labels}
            for i, object_label in enumerate( good_object_labels):
                for n, atuple in enumerate(list( itertools.product(var_list, range(len(func_set))))):
                    var = atuple[0]; k = atuple[1]
                    if 'min' in var:
                        min_var=True
                    else:
                        min_var=False
                    data = input_data[var]
                    # NE, NY, NX
                    amplitudes = self.ensemble_amplitudes( data = data, storm_points=storm_points, object_label=object_label, min_var=min_var)
                    features[i, n] = self._generic_function( func_set[k][0],
                                                   input_data=amplitudes,
                                                   parameter=func_set[k][1],
                                                  )

        return features, feature_names #dim = (n_objects, n_var*n_stats) 
 
        
    def ensemble_amplitudes(self, data, storm_points, object_label, min_var=False):
        '''
        ---------
        Args:
            data, array of shape (NE, NY, NX)
        '''
        y_set = storm_points[object_label][0]
        x_set = storm_points[object_label][1]
        if min_var:
            percentile = 10
        else:
            percentile = 90
        amplitudes = np.nanpercentile( data[:,y_set,x_set], percentile, axis=1 )  

        return amplitudes



    def _extract_storm_features_in_circle( self, input_data, x_object_cent, y_object_cent ): 
        ''' Extract intra-storm state features for machine learning '''
        x = np.arange(input_data.shape[-1])
        y = np.arange(input_data.shape[-2])
        stat_functions = self._set_of_stat_functions( )
        object_centroids = list(zip( y_object_cent, x_object_cent ))
        obj_strm_data = np.zeros(( len(object_centroids), input_data.shape[0] * len(stat_functions) ))
        for i, obj_cent in enumerate( object_centroids ):    
            rho, phi = self._cart2pol( x[np.newaxis,:]-obj_cent[1], y[:,np.newaxis]-obj_cent[0] )            
            storm_points = np.where(rho <= self.ROI_STORM )
            for j, k in enumerate( list(itertools.product( range(input_data.shape[0]), range(len(stat_functions)) ))):
                v = k[0] ; s = k[1]
                temp_data = input_data[v,:,:]
                func_set = stat_functions[s]
                obj_strm_data[i, j] = self._generic_function( func=func_set[0], input_data=temp_data[storm_points], parameter=func_set[1])
        
        return obj_strm_data #dim = (n_objects, n_var*n_stats)     

    def _extract_environment_features_in_arcregion( self, input_data, x_object_cent, y_object_cent, avg_bunk_v_per_obj, avg_bunk_u_per_obj ):
        ''' Extract storm-inflow environment features for machine learning '''
        x = np.arange(input_data.shape[-1])
        y = np.arange(input_data.shape[-2])
        stat_functions = self._set_of_stat_functions( )
        object_centroids = list(zip( y_object_cent, x_object_cent ))
        obj_env_data = np.zeros(( len(object_centroids), input_data.shape[0] * len(stat_functions) ))
        for i, obj_cent in enumerate( object_centroids ):    
            rho, phi = self._cart2pol( x[np.newaxis,:]-obj_cent[1], y[:,np.newaxis]-obj_cent[0] )
            bunk_u = avg_bunk_u_per_obj[i]
            bunk_v = avg_bunk_v_per_obj[i]
            env_points = self._find_storm_inflow_region( bunk_u, bunk_v, rho, phi )
            for j, k in enumerate( list(itertools.product( range(input_data.shape[0]), range(len(stat_functions)) ))):
                v = k[0] ; s = k[1]
                temp_data = input_data[v,:,:]
                func_set = stat_functions[s]
                obj_env_data[i, j] = self._generic_function( func=func_set[0], input_data=temp_data[env_points], parameter=func_set[1])
                
        return obj_env_data #dim = (n_objects, n_var*n_stats)         

    def extract_storm_patch( self, input_data, x_object_cent, y_object_cent ):
        ''' Extract the patches centered on the obj_centers '''
        object_centroids = list(zip( y_object_cent, x_object_cent ))
        storm_patches = np.zeros(( len(object_centroids), self.grid_size, self.grid_size, np.shape(input_data)[0] ))
        # print storm_patches.shape (n_objects, 24, 24, 13)
        for i, obj_cent in enumerate( object_centroids ):
            obj_y = obj_cent[0]
            obj_x = obj_cent[1]
            for v in range( np.shape(input_data)[0] ):
                storm_patches[i, :,:, v] = input_data[ v, obj_y-self.delta:obj_y+self.delta, obj_x-self.delta:obj_x+self.delta ]

        return storm_patches 


    def extract_patch(self, data, centers):
        '''
        Extract patches
        data (y,x,v)
        '''
        n_objects = len(list(centers))
        n_vars = data.shape[0]
        storm_patches = np.zeros((n_objects, n_vars, self.grid_size, self.grid_size))
        for i, pair in enumerate(centers):
            obj_y, obj_x=pair
            storm_patches[i,:,:,:] = data[:, obj_y-self.delta:obj_y+self.delta, obj_x-self.delta:obj_x+self.delta]

        return storm_patches


    def _find_storm_inflow_region( self, bunk_v, bunk_u, rho, phi  ):
        ''' Find storm inflow region using the average intra-storm bunker's motion vector '''
        # Bunker's motion in degrees
        left = ( np.arctan2( bunk_v, bunk_u ) * (180./np.pi) ) + 10.
        right = left - 110. 
        inflow_indices = np.where((phi <= left )& (phi >= right)&(rho <= self.ROI_ENV ))
        
        return inflow_indices 

    def _cart2pol( self, x, y ):
        ''' Converts from cartesian coordinates to polar coordinates ''' 
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x) * (180./np.pi)
        return(rho, phi) 

    def _remove_objects_near_boundary(self, x_obj_cent, y_obj_cent, NY, NX):
        """ Removes objects with centroid too close to the domain boundaries 
            The buffer zone is a combination of a static distance from the boundary and size of the storm path """
        xlims = np.arange(self.BUFFER, NX - self.BUFFER +1)
        ylims = np.arange(self.BUFFER, NY - self.BUFFER +1)

        good_idx = [ ]
        for i, centers in enumerate( list(zip(x_obj_cent, y_obj_cent))): 
            if (centers[1] in ylims and centers[0] in xlims):  
                good_idx.append( i )     
        
        return good_idx

    def _set_of_stat_functions( self, names=True, only_mean=False ):
        """ Function returns a list of function objects """
        func_set   = [  (np.nanmean,None), (np.nanpercentile, 10), (np.nanpercentile, 90) ]
        func_names = [  '__spatial_mean', '__spatial_10th',  '__spatial_90th' ]      
    
        if only_mean:
            if names:
                return [(np.nanmean, None)], [ '_spatial_mean']
            else:
                return [(np.nanmean, None)]
        else:
            if names:
                return func_set, func_names
            else:
                return func_set

    def _generic_function( self, func , input_data, parameter=None ):
        """ A function meant to implement the various function objects in 'set_of_stat_functions'"""
        if parameter == 'ddof':
            return func( input_data, ddof=1 )
        elif parameter is not None:
            return func( input_data, parameter)
        else:
            return func( input_data )




