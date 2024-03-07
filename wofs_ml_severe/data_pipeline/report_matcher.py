from monte_python.object_matching import match_to_lsrs, ObjectMatcher
import monte_python
from .storm_report_loader import StormReportLoader
from .warning_polygon_loader import WarningPolygonLoader
from ..common.util import get_init_time, get_valid_time

import xarray as xr 
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from scipy.ndimage import maximum_filter
import traceback
from os.path import basename, dirname
from datetime import timedelta, datetime 
import gc 
from pathlib import Path 


class MatchToTracks:
    """
    Match local storm reports, MESH, warning polygons and other
    observations to the ensemble storm track objects using object matching.
    
    The storm reports are converted to a gridded, labeled 2D array to perform the
    object matching. This class produces the MLTARGETS dataframe associated with
    an ENSEMBLETRACKS summary file.
    
    Attributes:
        min_dists (list of int): Minimal distances for object matching (grid units).
        err_window (int): Error window in minutes (defaults to 15). Allows for loading
                          reports within the valid time window considering this error.
        return_df (bool): If True, return results dataframe without altering MLTARGETS.
        reports_path (str): Path to the storm events dataframe.
        forecast_length (int): Length of forecast period (in minutes)
        size (int): Diameter of the maximum value filter (in gridpoints)
        n_expected_files (int): Number of MRMS files for the forecast length (default=13)
                                If at least half of the files are missing for a forecast length,
                                then the data are flagged. 
    """
    
    RATINGS_MAP = {'EF0': 1, 'EF1': 2, 'EF2': 3, 'EF3': 4, 'EF4': 5, 'EF5': 6}
    
    MRMS_PATH = '/work/rt_obs/MRMS/RAD_AZS_MSH/'
    MRMS_PATHS = {
              '2018' : '/work/brian.matilla/WoFS_2020/MRMS/RAD_AZS_MSH/2018/',
              '2019': '/work/brian.matilla/WoFS_2020/MRMS/RAD_AZS_MSH/2019/',
              '2020' : '/work/brian.matilla/WoFS_2020/MRMS/RAD_AZS_MSH/2020/',
              '2021' : '/work/brian.matilla/WOFS_2021/MRMS/RAD_AZS_MSH/',
              '2022' : '/work/rt_obs/MRMS/RAD_AZS_MSH/2022/',
             }
    
    # 1 in == 25.4 mm 
    #IN_TO_MM = 25.4
    
    THRESH = 1.0 #25.4
    THRESH_30MM = 1.1811023622047 # 30MM
    
    def __init__(self, reports_path, min_dists=None, err_window=15, return_df=False, 
                forecast_length=30, size=3, n_expected_files=13, verbose=False):
                 
        if min_dists is None:
            min_dists = [0, 1, 2, 5, 10]
        if reports_path is None:
            reports_path = '/work/mflora/LSRS/STORM_EVENTS_2017-2023.csv'
        
        self._min_dists = min_dists
        self.err_window = err_window
        self.return_df = return_df
        self.reports_path = reports_path
        self.size = size
        self.forecast_length=forecast_length
        self.n_expected_files = n_expected_files
        
        self._magnitude_keyword = 'any'
        self.match_to_reports = False
        self.verbose=verbose

    def __call__(self, track_file):
        
        results = self._load_wofs_tracks(track_file)
        if results is None:
            return None 
        
        tracks, labels = results 
        
        if self.verbose:
            print('Starting the target file building processing...') 
        
        # ******* LOCAL STORM REPORTS *************
        # Match to the storm reports 
        if self.verbose:
            print('Matching to reports...') 
        lsr_dataset, lsr_points = self.get_reports(track_file)
        lsr_df = self.object_match(tracks, labels, lsr_dataset, one_to_one=False, 
                     obs_points=lsr_points)
        
        # ******* MRMS MESH *************
        # Load MESH data and identify tracks. 
        if self.verbose:
            print('Matching to MESH...') 
        mesh_arr = self.load_mrms(track_file)
                    
        # If the MESH doesn't load, return empty data
        if mesh_arr is None:
            targets_data={}
            targets_data['max_mesh'] = [-1]*len(labels)
            for var in ['mesh_severe', 'mesh_sig_severe', 'mesh_severe_30mm']: 
                for min_d in self._min_dists:
                    # Create target column 
                    targets_data[f"{var}_{min_d*3}km"] = [-1]*len(labels)
                
            mrms_df = pd.DataFrame(targets_data)
            
        else:
            # TODO: add sig hail MESH
            mesh_svr, _ = monte_python.label( input_data = mesh_arr,
                                   method ='single_threshold', 
                                   return_object_properties=True, 
                                   params = {'bdry_thresh': self.THRESH } )
        
            mesh_svr_30mm, _ = monte_python.label( input_data = mesh_arr,
                                   method ='single_threshold', 
                                   return_object_properties=True, 
                                   params = {'bdry_thresh': self.THRESH_30MM } )
        

            mesh_sig_svr, _ = monte_python.label( input_data = mesh_arr,
                                   method ='single_threshold', 
                                   return_object_properties=True, 
                                   params = {'bdry_thresh': 2*self.THRESH_30MM } )
        
            # Combine into xarray dataset
            # Define coordinates and dimensions
            coords = {'NX': np.arange(tracks.shape[1]), 'NY': np.arange(tracks.shape[0])}
            dims = ('NX', 'NY')

            data_dict = {'mesh_severe' : mesh_svr,
                         'mesh_severe_30mm': mesh_svr_30mm, 
                         'mesh_sig_svr': mesh_sig_svr
            }
            
            
            # Use list comprehension to create a list of DataArray objects
            data_arrays = [xr.DataArray(data=arr, coords=coords, dims=dims, 
                                        name=name) for name, arr in data_dict.items()]

            # Combine the DataArrays into a Dataset
            mrms_dataset = xr.Dataset({da.name: da for da in data_arrays})

            mrms_df = self.object_match(tracks, labels, mrms_dataset, one_to_one=False)
            mrms_df['max_mesh'] = self.extract_spatial_amplitude(mesh_arr, tracks, labels)

        # ******* NWS WARNING POLYGONS *************
        # Load warnings polygons  TODO
        if self.verbose:
            print('Matching to NWS polygons...') 
        
        polygon_loader = WarningPolygonLoader(track_file, self.forecast_length, self.err_window)
        polygon_dataset = polygon_loader.load()
        poly_df = self.object_match(tracks, labels, polygon_dataset, one_to_one=False)
        
        # TODO: Append dfs together! 
        if self.verbose:
            print('Combining the dataframes...') 
            
        final_df = pd.concat([lsr_df, mrms_df, poly_df], axis=1)
     
        final_df['labels'] = labels
    
        # Save the data to the MLTARGETS.
        if self.return_df:
            return final_df
        
        target_file = self._save_targets(final_df, track_file)  
        
        return target_file    
    
    def get_reports(self, ncfile, report_type='NOAA', size=3, forecast_length=30, to_xy=True,
                    prob_thresholds=None):
        """
        Get the storm reports for the forecast period.
        
        Parameters:
            ncfile (str): Path to the netCDF file.
            report_type (str): Type of report, default is 'NOAA'.
            size (int): Size for grid mapping.
            forecast_length (int): Forecast period length in minutes.
            to_xy (bool): Whether to convert to x and y coordinates.
            prob_thresholds (numpy.ndarray): Probability thresholds for reports.
        
        Returns:
            Tuple containing gridded dataset and list of storm report points.
        """
        if prob_thresholds is None:
            prob_thresholds = np.arange(0.05, 0.8, 0.1)
        
        init_time = get_valid_time(ncfile)
        report = StormReportLoader(
            self.reports_path,
            report_type,
            init_time,
            forecast_length=forecast_length,
            err_window=self.err_window
        )
        
        try:
            ds = xr.load_dataset(ncfile, decode_times=False)
        except OSError as e:
            print(f'Error loading {ncfile}: {e}')
            return None
        
        try:
            lsr_points = report.get_points(
                dataset=ds, magnitude=self._magnitude_keyword, to_xy=to_xy
            )
            if not to_xy:
                return lsr_points
            
            grid_ds = report.to_grid(dataset=ds, points=lsr_points, size=size)
        except Exception as e:
            print(f'Error during report processing: {traceback.format_exc()}')
            return None
        
        return grid_ds, lsr_points

    def _load_wofs_tracks(self, track_file):
        try:
            tracks_ds = xr.load_dataset(track_file, decode_times=False)
        except Exception as e:
            print(f'Error loading {track_file}: {e}')
            return None
    
        tracks = tracks_ds['w_up__ensemble_tracks'].values
        #ens_probs = tracks_ds['w_up__ensemble_probabilities'].values
    
        # Skip the unique function if there are no tracks
        if not np.any(tracks):
            return None

        labels = np.unique(tracks)[1:]
    
        return tracks, labels 
    
    def dt_rng(self):
        # Convert string to datetime object
        sdate = datetime.strptime(self.sdate, '%Y%m%d%H%M')
        edate = sdate+timedelta(minutes=self.forecast_length)+timedelta(minutes=self.err_window)
        sdate-=timedelta(minutes=self.err_window)
    
        return sdate, edate 

    def find_mrms_files(self):
        """
        When given a start and end date, this function will find any MRMS RAD 
        files between those time periods. It will check if the path exists. 
        """
        sdate, edate = self.dt_rng()
        date_rng = pd.date_range(sdate, edate, freq=timedelta(minutes=5))
        
        mrms_filenames = [date.strftime('wofs_MRMS_RAD_%Y%m%d_%H%M.nc') for date in date_rng]
        mrms_filepaths = [Path(self.MRMS_PATH).joinpath(self.year, self.date, f) for f in mrms_filenames 
                  if Path(self.MRMS_PATH).joinpath(self.year, self.date, f).is_file()
                 ]
    
        return mrms_filepaths 
    
    def load_mrms(self, ncfile):
        # Load the file for the 30-min+ and concate along time dim.
        
        # Get the beginning of the 30-min period for the ENSEMBLETRACK file.
        self.sdate = get_valid_time(ncfile)
        self.year = self.sdate[:4]
        self.date = Path(ncfile).parent.parent.stem
        
        try:
            files = self.find_mrms_files()
        except Exception as e:
            print(f'Issues finding MRMS MESH files for {self.sdate}! {traceback.format_exc()}')
            return None
        
        # Check if at least half the expected files exist.
        if len(files) <= int(self.n_expected_files/2):
            print(f'Half of the files are missing for {self.sdate}!')
            return None
        
        try:    
            # Initialize an empty list to store the datasets with 'mesh_consv' variable
            datasets = []

            # Load 'mesh_consv' variable from each file and append to the datasets list
            for file in files:
                ds = xr.open_dataset(file, drop_variables=['lat', 'lon'])
                datasets.append(ds['mesh_consv'].values)
                ds.close()
                
            # Concatenate the datasets along the 'time' dimension
            max_vals = np.max(datasets, axis=0)
            
            # Replace NaN values with zeros. 
            max_vals[np.isnan(max_vals)] = 0
        except Exception as e:
            print(f'Issues loading files for {self.sdate}: {files}. List is likely empty. {traceback.format_exc()}')
            return None

        return max_vals 
    
    def extract_spatial_amplitude(self, obs_arr, forecast_tracks, track_labels ):
        """Extract the spatial maximum value from some field from data inside a track.
        A spatial maximum value filter is first applied to capture spatial uncertainty."""
        max_obs_arr = maximum_filter(obs_arr, self.size)
        data = [np.max(max_obs_arr[np.where(forecast_tracks==label)]) 
                              for label in track_labels] 
            
        return data
    
    def object_match(self, forecast_tracks, track_labels, obs_dataset, one_to_one=False, 
                     obs_points=None):
        """
        Generic function for object matching between a set of forecast tracks
        and either a 2D observation dataset or set of points (e.g., local storm reports)
        
        Parameters
        ---------------------
        track_file: path: Path to an ENSEMBLETRACKS summary file.  
        
        """
        target_variables= obs_dataset.data_vars
        if target_variables is None:
            return None
        
        targets_data = {}
        vals=None
        for var in target_variables:
            target = obs_dataset[var].values
            if obs_points is not None:
                these_points = obs_points[var]
                # The points are (lon, lat, val) for LSRs. 
                vals = np.array([v[-1] for v in these_points])
            
            for min_d in self._min_dists:
                try:
                    obj_match = ObjectMatcher(min_dist_max=min_d, 
                                              score_thresh=0.2, 
                                              time_max=0, 
                                              one_to_one=one_to_one, 
                                              match_to_reports=self.match_to_reports)
                    
                    matched_tracks, matched_obs, _ = obj_match.match_objects(
                                                                        object_set_a=forecast_tracks, 
                                                                        object_set_b=target)
                    
                    matched_obs = np.array(matched_obs)
                    unique_matched_tracks = np.unique(matched_tracks)
                    target_arr = self._construct_target_array(track_labels, unique_matched_tracks, 
                                                          matched_tracks, matched_obs, vals, var)
                    
                except Exception as e:
                    print(f'Error during object matching: {traceback.format_exc()}')
                    target_arr = [-1]*len(track_labels) 
                
                targets_data[f"{var}_{min_d*3}km"] = target_arr
                
        df = pd.DataFrame(targets_data)
        
        return df 

    def _construct_target_array(self, labels, unique_matched_tracks, matched_tracks, matched_reports, vals, var):
        """
        Construct the target array for a given variable.
        """
        if vals is None:
            # Create target column 
            target_arr =  [1 if label in matched_tracks else 0 for label in labels]
            return target_arr
        
        target_arr = []
        for label in labels:
            if label in unique_matched_tracks:
                inds = np.where(matched_tracks == label)[0]
                report_inds = matched_reports[inds] - 1
                all_values = vals[report_inds]
                val_to_keep = max(all_values,
                  key=lambda r: self.RATINGS_MAP.get(r, 0)) if 'tornado' in var else np.max(all_values)
                if 'tornado' in var: 
                    val_to_keep = self.RATINGS_MAP.get(val_to_keep, 0)
                
                
                target_arr.append(val_to_keep)
            else:
                target_arr.append(0)
        return target_arr

    def _save_targets(self, final_df, track_file):
        """
        Save the final dataframe to a file and perform cleanup.
        """
        target_file = track_file.replace('ENSEMBLETRACKS', 'MLTARGETS').replace('.nc', '.feather')
        final_df.to_feather(target_file)
        gc.collect()
        return target_file