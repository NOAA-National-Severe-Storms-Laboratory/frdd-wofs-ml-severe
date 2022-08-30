#======================================================
# A complete data pipeline for generating the ML datasets
# used to train and evaluate the WoFS-ML-Severe products.
# 
# Author: Montgomery Flora (Git username : monte-flora)
# Email : monte.flora@noaa.gov 
#======================================================

# Python Modules 
from os.path import join, exists
import os 
from glob import glob 
import datetime
import itertools 
import traceback
import logging 

# Third-party modules
import pandas as pd 
import xarray as xr 
import numpy as np 
from skimage.measure import regionprops

# Personal Modules
from ..common.util import Emailer
from ..common.multiprocessing_utils import run_parallel, to_iterator

# WoF_post modules 
from WoF_post.wofs.ml.wofs_ensemble_track_id import generate_ensemble_track_file
from WoF_post.wofs.ml.wofs_ml_severe import MLOps

# Will need to make sure this is update!!
from monte_python.object_matching import match_to_lsrs, ObjectMatcher

# TODO: Update the StormReporter and point to it appropriately!!
from WoF_post.wofs.verification.lsrs.get_storm_reports import StormReports
from WoF_post.wofs.plotting.util import decompose_file_path
from WoF_post.wofs.post.utils import save_dataset

class Logger:
    def __init__(self, filename='data_pipeline.log'):
        logFormat = '%(asctime)s...%(levelname)s %(message)s'
        logging.basicConfig(filename=filename,
                            filemode='a',
                            format=logFormat,
                            datefmt="%d-%b %H:%M",
                            level = logging.INFO
                            )
        self.logger = logging.getLogger()
   
    def __call__(self, log_type, message, **kwargs):
        """ log info, error, debug, or critical message to a log file """
        getattr(self.logger, log_type)(message, **kwargs)

class MLDataPipeline(Emailer):
    """
    DataPipeline maintains the data pipeline for generating the data 
    used to train and evaluate the WoFS-ML-Severe product. 
    
    The data pipeline is as follows:
    
    1. Identify ensemble tracks.
        - Checks if the file already exists.
        
    2. Perform the feature engineering, build the dataframes, save it 
        - Checks if the file already exists. 
        
    3. Get the most up-to-date storm reports or storm data. 
        
    4. Match the ensemble storm tracks to storm reports.
        - storm reports are converted to grids 
        
    5. Concatenate the dataframe together with the target dataframes
    
    """
    METADATA = ['forecast_time_index', 'Run Date', 'Initialization Time', 'obj_centroid_x', 
            'obj_centroid_y', 'label']
    
    # TODO: Add the ensemble storm tracks parameters as an arg for data pipeline.
    # Then add the ensemble storm track parameters to a config file. 
    
    def __init__(self, dates=None, times = ['2200'], n_jobs=30, out_path ='/work/mflora/ML_DATA/DATA/', verbose=True):
        self.logger = Logger()
        self._base_path = '/work/mflora/SummaryFiles'
        self.out_path = out_path 
        
        if dates is None:
            self.dates = [d for d in os.listdir(self._base_path) if '.txt' not in d]
            self.send_email_bool = True
        else:
            self.dates = dates
            self.times = times 
            self.send_email_bool = True
        
        self._runtype = 'rto'
        self.n_jobs = n_jobs
        
        # Variables required for the summary file names. 
        self._DT = 5 
        self._DURATION = 30 
        self._NT = 36 
        
        
    def __call__(self, delete_existing=False):
        """ Initiates the date building."""
        # TO-DO: if delete_existing, then delete any 
        # existing ENSEMBLETRACKS, MLTARGETS, MLDATA
        
        self.logger('info', '='*50) 
        self.logger('info', '============= STARTING A NEW DATA PIPELINE =============') 
        
        # Identify the ensemble storm tracks. 
        self.logger('info', '========== IDENTIFYING THE ENSEMBLE STORM TRACKS =======') 
        self.get_ensemble_tracks()
        
        # Extract the ML features from the ensemble storm tracks.
        self.logger('info', '======== EXTRACTING THE ML FEATURE USING THE TRACKS =====') 
        self.get_ml_features()
        
        # Match to the storm reports.
        self.logger('info', '============ MATCHING TRACKS TO STORM REPORTS ===========') 
        self.match_to_storm_reports()
        
        # Concatenate data together and create a single dataframe.
        # Also, appends the target dataframes.
        self.logger('info', '============ BUILDING THE FINAL DATASETS ===========') 
        self.concatenate_dataframes()
    
    def get_ensemble_tracks(self,):
        """ Identifies the ensemble tracks from the 30M files """
        # TODO: Add some kind of config to set the parameters of the object ID. 
        # TODO: Add the ability to override existing data in an ensemble storm track.
        
        # Get the start time, which is used for computing the 
        # compute duration. 
        start_time = self.get_start_time()
        
        # Get the filenames. 
        filenames = self.get_files_for_tracks()
        
        # Identify the ensemble storm tracks (using multiprocessing). 
        if len(filenames) > 0:
            run_parallel(
                func = generate_ensemble_track_file,
                nprocs_to_use = self.n_jobs,
                iterator = to_iterator(filenames),
                kwargs = {'logger' : self.logger}, 
                )
            message = "Re-processing of the Ensemble Storm Track files is complete!"
        else:
            message = "No Ensemble storm tracks to process, moving onto the next step...."
            
        # Send an email to myself once the process is done! 
        if self.send_email_bool:
            self.send_message(
                               message, start_time)
        
    def get_ml_features(self,):
        """ Extract ML features from the WoFS using the 
        ensemble storm tracks"""
        # Get the summary file path directories.
        indirs = self.get_files(file_type=None)
     
        files_to_load = []
        path = os.getcwd()
        for indir in indirs:
            year = indir.split('/')[-2][:4]
            if year == '2017':
                ml_config_path = join(path, 'ml_config_2017.yml')
            elif year in ['2018', '2019']:
                ml_config_path = join(path, 'ml_config_2018-19.yml')
            else:
                ml_config_path = join(path, 'ml_config.yml')
            
            mlops = MLOps(TEMP=False, test=True, ml_config_path=ml_config_path, logger=self.logger)    
            
            try:
                _files_to_load = self.get_files_for_ml(indir)
            except:
                self.logger('error', f'Files were not available or had issues for {indir}')
                self.logger('critical', traceback.format_exc()) 
                _files_to_load = []
                
            for f in _files_to_load:
                ml_file = f['track_file'].replace('ENSEMBLETRACKS', 'MLDATA').replace('.nc', '.feather') 
                
                if not exists(ml_file):
                    files_to_load.append(f)
                else:
                    self.logger('debug', f'{ml_file} already exists!...')  
                    
        start_time = self.get_start_time()

        if len(files_to_load) > 0:
            # MAIN FUNCTION
            try:
                mlops(n_processors=self.n_jobs, 
                  runtype=self._runtype,
                  predict=False,
                  files_to_load=files_to_load
                 ) 
            except:
                print(traceback.format_exc())
                
        # Check that for every ENSEMBLETRACK file, there is a corresponding MLDATA file! 
        #files = self.files_to_run(original_type='MLDATA', new_type = 'ENSEMBLETRACKS')
        #print('ml_files:', files) 
        #if len(files) > 1:
        #    self.logger('critical', 'Not all MLDATA files were created for the existing ENSEMBLETRACK files')
        
            if self.send_email_bool:
                self.send_message("ML feature extraction is finished!", start_time)  
        
    def match_to_storm_reports(self, 
                               cent_dist_max=15.0, 
                               time_max=0, 
                               score_thresh=0.2, 
                               dists=[1,3,5,10]):
        """ Match ensemble storm tracks to storm reports 
        
        Parameters
        --------------------
        cent_dist_max
        time_max
        score_thresh
        dists 
        
        """
        start_time = self.get_start_time()
        
        # Get the filenames. 
        filenames = self.files_to_run(original_type='ENSEMBLETRACKS', new_type = 'MLTARGETS')
        
        ###print(f'{filenames=}')
              
        # TODO: incorporate the new matching!!!    
            
        def worker(track_file):
            """
            Match the ensemble storm tracks to the gridded LSRs. 
            Outputs a dataframe of targets for the MLDATA-based summary files.
    
            Multiple matching minimum matching distances are used. 
            """
            try:
                tracks_ds = xr.load_dataset(track_file, decode_times=False)
                tracks = tracks_ds['w_up__ensemble_tracks'].values
                object_props = regionprops(tracks, tracks)
                labels = np.unique(tracks)[1:]
                storm_data_ds, lsr_points = self.reports_to_grid(track_file)
    
                target_vars = [v for v in storm_data_ds.data_vars if 'severe' in v] 
    
                if len(target_vars) == 0:
                    return None 
    
                target_obj_match_dict = {}
                target_old_dict = {}
            
                for var in target_vars:
                    target = storm_data_ds[var].values
                    for min_dist_max in dists:
                        obj_match = ObjectMatcher(min_dist_max=min_dist_max,
                                      cent_dist_max=cent_dist_max,
                                      time_max=time_max,
                                      score_thresh = score_thresh,
                                      one_to_one = True)
                        
                        matched_tracks, _ , _ = obj_match.match_objects(object_set_a=tracks, object_set_b=target,)
                        match_dict = match_to_lsrs(object_props, lsr_points[var], dist_to_lsr=min_dist_max)
                        
                        # Create target column 
                        target_obj_match_dict[f"{var}_{min_dist_max*3}km_obj_match"] = [1 if label in 
                                                                              matched_tracks else 0 for label in labels]
                        
                        target_old_dict[f"{var}_{min_dist_max*3}km"] = [match_dict[l] for l in labels]
                         
                full_target_dict = {**target_obj_match_dict, **target_old_dict}       

                storm_data_ds.close()
                tracks_ds.close()
            
                del tracks_ds, storm_data_ds
            
                df = pd.DataFrame(full_target_dict)
                target_file = track_file.replace('ENSEMBLETRACKS', 'MLTARGETS').replace('.nc', '.feather')
            
                self.logger('debug', f'Saving {target_file}...') 
                df.to_feather(target_file)
    
                return None 
            except:
                print(traceback.format_exc())
                return None 
                
        if len(filenames) > 0:
    
            run_parallel(
                func = worker,
                nprocs_to_use = self.n_jobs,
                iterator = to_iterator(filenames),
                mode='joblib',
                )
        
            if self.send_email_bool:
                self.send_message('Matching to storm reports is finished!', start_time)
            
    def reports_to_grid(self, ncfile,):
        """ Converts storm reports to a grid for object matching. """
        
        # This code should be inside the StormReporter
        
        # Determine the initial time from the ncfile 
        comps = decompose_file_path(ncfile)
        init_time = comps['VALID_DATE']+comps['VALID_TIME']
        report = StormReports(init_time, 
            forecast_length=30,
            err_window=15, 
            )
 
        ds = xr.load_dataset(ncfile)
        
        try:
            grid_ds = report.to_grid(dataset=ds)
            lsr_points = report.get_points(dataset=ds)
            
        except Exception as e:
            self.logger('info', f'Unable to process storm reports for {ncfile}!')
            self.logger('error', e, exc_info=True) 
            
        return grid_ds, lsr_points
    
    def concatenate_dataframes(self,):
        """ Load the ML features and target dataframes
        and concatenate into a single dataframe """
        delta_time_step = int(self._DURATION / self._DT)
        start_time = self.get_start_time()
        
        ml_files = self.get_files('MLDATA')
        target_files = [f.replace('MLDATA', 'MLTARGETS') for f in ml_files] 
        
        dfs = [pd.read_feather(f) for f in ml_files]
        
        feature_df = pd.concat(dfs)
        target_df = pd.concat([pd.read_feather(f) for f in target_files])
        
        forecast_time_index = [int(decompose_file_path(f)['TIME_INDEX']) - delta_time_step for f in ml_files]
        forecast_time_index = [[ind]*len(_df) for ind, _df in zip(forecast_time_index, dfs)] 
        forecast_time_index = [item for sublist in forecast_time_index for item in sublist]
        
        df = pd.concat([feature_df, target_df], axis=1) 
        df['forecast_time_index'] = np.array(forecast_time_index, dtype=np.int8) 
        
        ranges = [np.arange(13), np.arange(12,37)]
        names = ['first_hour', 'second_hour']
        
        for name, rng in zip(names, ranges): 
            # Get the examples within a particular forecast time index range.
            _df = df[df['forecast_time_index'].isin(rng).values].reset_index(drop=True) 
            
            baseline_features = [f for f in _df.columns if '__prob_max' in f]
            targets = [f for f in _df.columns if 'severe' in f]

            baseline_df = _df[baseline_features+self.METADATA+targets]

            ml_features = [f for f in _df.columns if f not in baseline_features]
            ml_df = _df[ml_features]
           
            baseline_df.to_feather(join(self.out_path, f'wofs_ml_severe__{name}__baseline_data.feather'))
            ml_df.to_feather(join(self.out_path, f'wofs_ml_severe__{name}__data.feather'))

        if self.send_email_bool:
            self.send_message('Final datasets are built!', start_time)
    
    def get_files(self, file_type=None):
        """
        Returns a list of filename with the given file type.
        """
        paths = []
        if file_type is None:
            # When no file type is given, just return the file directories. 
            for d in self.dates:
                times = os.listdir(join(self._base_path, d))
                dirs = [join(self._base_path,d,t) for t in times]
                paths.extend(dirs)
        else:
            for d in self.dates:
                path = join(self._base_path,d)
                for (dir_path, _, file_names) in os.walk(path):
                    file_names = [join(dir_path, f) for f in file_names if file_type in f]
                    paths.extend(file_names) 
            
        return paths 
    
    def files_to_run(self, original_type='30M', new_type = 'ENSEMBLETRACKS'):
        """ Checks that files are created! """
        files_to_create = [] 
        
        files = self.get_files(original_type)
        
        if new_type in ['MLDATA', 'MLTARGETS']:
            new_files = [f.replace(original_type, new_type).replace('.nc', '.feather') for f in files]
        else:
            new_files = [f.replace(original_type, new_type) for f in files]
        all_exist = [exists(f) for f in new_files]
        if not all(all_exist):
            inds = [i for i, x in enumerate(all_exist) if not x]
            for i in inds:
                self.logger('debug', f"{new_files[i]} does not exist, but {files[i]} does!") 
                files_to_create.append(files[i]) 
        
        return files_to_create
    
    def get_files_for_ml(self, indir):
        """ Get the summary files for the ML feature extraction """
        files_to_load = [] 
        delta_time_step = int(self._DURATION / self._DT)
        
        for t in range((self._NT-delta_time_step)+1):
            try:
                track_file = glob(join(indir, f'wofs_ENSEMBLETRACKS_{t+delta_time_step:02d}*'))[0]
                
                env_file = glob(join(indir, f'wofs_ENV_{t:02d}*'))[0]
                svr_file = env_file.replace('ENV', 'SVR')
                ens_files = [glob(join(indir, f'wofs_ENS_{_t:02d}*'))[0] for _t in range(t, t+delta_time_step+1)]                
                files_to_load.append( {'track_file': track_file, 
                         'ens_file' : ens_files, 
                         'env_file'  : env_file, 
                         'svr_file'  : svr_file,      
                        }) 
            
            except IndexError:
                print(f'IndexError! {indir}/ENSEMBLETRACK file at {t+delta_time_step} is not available!') 
    
        return files_to_load
    
    def get_files_for_tracks(self,):
        """ Get the 30M files for the ensemble storm tracks """
        # Get the filenames. 
        filenames = self.files_to_run(original_type='30M', new_type = 'ENSEMBLETRACKS') 
        
        # Only processing files within the first three hours. 
        filenames = [f for f in filenames if int(decompose_file_path(f)['TIME_INDEX']) <= self._NT]
        
        return filenames 
    