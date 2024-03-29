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
from pathlib import Path

# Third-party modules
import pandas as pd 
import xarray as xr 
import numpy as np 
from skimage.measure import regionprops
from tqdm import tqdm 
import random

# Personal Modules
from ..common.emailer import Emailer
from ..common.multiprocessing_utils import run_parallel, to_iterator
from ..common.util import decompose_file_path, save_dataset
from ..io.io import MLDataLoader

from .ensemble_track_segmentation import generate_ensemble_track_file
from .ml_data_generator import MLDataGenerator
from .report_matcher import MatchToTracks
from .concatenator import Concatenator

class Logger:
    def __init__(self, filename='data_pipeline.log'):
        logFormat = '%(asctime)s...%(levelname)s %(message)s'
        logging.basicConfig(filename=filename,
                            filemode='a',
                            format=logFormat,
                            datefmt="%d-%b %H:%M",
                            level = logging.INFO
                            )
        #print = logging.getLogger()
   
    def __call__(self, log_type, message, **kwargs):
        """ log info, error, debug, or critical message to a log file """
        getattr(print, log_type)(message, **kwargs)

class MLDataPipeline(Emailer):
    """
    DataPipeline maintains the data pipeline for generating the data 
    used to train and evaluate the WoFS-ML-Severe product. 
    
    The data pipeline is as follows:
    
    1. Identify ensemble tracks.
        - Checks if the file already exists.
        
    2. Perform the feature engineering, build the dataframes, save them 
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
    
    def __init__(self, dates=None, times = None, previous_method=False,
                 n_jobs=30, out_path ='/work/mflora/ML_DATA/DATA/', verbose=True):
        
        self._BASE_PATH = '/work/mflora/SummaryFiles'
        self.reports_path = '/work/mflora/LSRS/STORM_EVENTS_2017-2023.csv'
        self.out_path = out_path 
        self.verbose=verbose
        self.fix_date = True
        
        if dates is None:
            # TODO: Make it year based! 
            ##self.dates = [d.split('_')[0] for d in os.listdir(self._BASE_PATH) if '.txt' not in d]
            
            possible_dates = [d for d in os.listdir(self._BASE_PATH) if '.txt' not in d and 'old' not in d]
            possible_dates = [d for d in possible_dates if  8 <= len(d) <= 11 ]

            possible_dates.sort()

            valid_years = [2018, 2019, 2020, 2021, 2022]#, 2023]
            self.dates = [date for date in possible_dates if int(date[:4]) in valid_years]
            
            self.send_email_bool = True
            self.times=None
            self._NT = 36
            self.debug=False
        else:
            self.dates = dates
            self.times = times
            self.sample_size = 18 
            self.debug = True
            self._NT = 2
            self.send_email_bool = False
        
        self._runtype = 'rto'
        self.n_jobs = n_jobs
        
        # Variables required for the summary file names. 
        self._DT = 5 
        self._DURATION = 30
        self._previous_method = previous_method
        
    def __call__(self, skip=[], delete_types=['MLDATA', 'ENSEMBLETRACKS', 'MLTARGETS', 'FINAL'], 
                keep_existing_ml_files=False):
        """ Initiates the date building."""
        print(f'Deleting existing files {delete_types}...')
        self.delete_existing_files(delete_types)
        
        self.keep_existing_ml_files = keep_existing_ml_files
        
        if 'FINAL' in delete_types:
            os.system('rm /work/mflora/ML_DATA/DATA/wofs*')
        
        print('info', '='*50) 
        print('info', '============= STARTING A NEW DATA PIPELINE =============') 
        
        # Identify the ensemble storm tracks. 
        if 'get_ensemble_tracks' not in skip:
            print('info', '========== IDENTIFYING THE ENSEMBLE STORM TRACKS =======')
            self.get_ensemble_tracks()
        
        # Extract the ML features from the ensemble storm tracks.
        if 'get_ml_features' not in skip:
            print('info', '======== EXTRACTING THE ML FEATURE USING THE TRACKS =====') 
            self.get_ml_features()
        
        if 'append_ml_features' not in skip:
            self.append_ml_features()
        
        # Match to the storm reports.
        if  'match_to_storm_reports' not in skip:
            print('info', '============ MATCHING TRACKS TO STORM REPORTS ===========')
            self.match_to_storm_reports()
        
        # Concatenate data together and create a single dataframe.
        # Also, appends the target dataframes.
        print('info', '============ BUILDING THE FINAL DATASETS ===========') 
        self.concatenate_dataframes()
    
    def delete_existing_files(self, types):
        """Delete existing files"""
        if len(types)==0:
            return None 
        
        base_path = '/work/mflora/SummaryFiles'

        paths = []
        date_paths = [join(self._BASE_PATH, d) for d in os.listdir(base_path)]
        for path in date_paths:
            for (dir_path, _, file_names) in os.walk(path):
                file_names = [join(dir_path, f) for f in file_names if any([t in f for t in types])]
                paths.extend(file_names)
        
        removal = [os.remove(p) for p in paths]
        
    def get_ensemble_tracks(self,):
        """ Identifies the ensemble tracks from the 30M files """
        # TODO: Add some kind of config to set the parameters of the object ID. 
        
        # Get the start time, which is used for computing the 
        # compute duration. 
        start_time = self.get_start_time()
        
        # Get the filenames.
        filenames = self.files_to_run(original_type='30M', new_type = 'ENSEMBLETRACKS') 

        # Identify the ensemble storm tracks (using multiprocessing). 
        if len(filenames) > 0:
            run_parallel(
                func = generate_ensemble_track_file,
                nprocs_to_use = self.n_jobs,
                args_iterator = to_iterator(filenames),
                kwargs = {'logger' : print},
                description='Generating Ensemble Storm Tracks'
                )
            message = "Re-processing of the Ensemble Storm Track files is complete!"
        else:
            message = "No Ensemble storm tracks to process, moving onto the next step...."
            
        # Send an email to myself once the process is done! 
        if self.send_email_bool:
            self.send_email(message, start_time)
        
    def get_ml_features(self):
        """ Extract ML features from the WoFS using the 
        ensemble storm tracks"""
        paths = self.get_files_for_ml()
    
        start_time = self.get_start_time()

        if len(paths) > 0:
            # MAIN FUNCTION
            try:
                mlops = MLDataGenerator(TEMP=False, retro=True, logger=print) 
                
                mlops(paths, 
                      n_processors=self.n_jobs, 
                      realtime=False)
            except:
                print(traceback.format_exc())
                
        if self.send_email_bool:
            self.send_email("ML feature extraction is finished!", start_time)  
    
    def append_ml_features(self,):
        """ Extract ML features from the WoFS using the 
        ensemble storm tracks. Appending onto existing dataframes.
        This will change in the future. 
        """
        paths = self.get_files("MLDATA")
        
        paths = [{'track_file' : f.replace('MLDATA', 'ENSEMBLETRACKS').replace('.feather', '.nc')}
                  for f in paths]
        
        start_time = self.get_start_time()

        if len(paths) > 0:
            # MAIN FUNCTION
            try:
                mlops = MLDataGenerator(TEMP=False, retro=True, logger=print) 
                
                mlops(paths, 
                      n_processors=self.n_jobs, 
                      realtime=False, append=True)
            except:
                print(traceback.format_exc())
                
        if self.send_email_bool:
            self.send_email("Appending ML feature extraction is finished!", start_time)  
    
    def match_to_storm_reports(self,):
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
        
        if self.debug: 
       
            # Randomly sample 'sample_size' filenames from the list
            filenames = random.sample(filenames, self.sample_size)
        
        if len(filenames) > 0:
            run_parallel(
                func = MatchToTracks(self.reports_path, 
                                     min_dists=[0,1], 
                                     err_window=15, 
                                     return_df=False, 
                                     forecast_length=30, 
                                     size=3, 
                                     n_expected_files=13,
                                     verbose=False
                                    ),
                nprocs_to_use = self.n_jobs,
                args_iterator = to_iterator(filenames),
                description='Matching Reports to Tracks'
                )
        
            if self.send_email_bool:
                self.send_email('Matching to storm reports is finished!', start_time)
    
    def concatenate_dataframes(self,):
        """ Load the ML features and target dataframes
        and concatenate into a single dataframe """
        start_time = self.get_start_time()
        ml_files = self.get_files('MLDATA')

        Concatenator()(ml_files, self.out_path, fix_date=self.fix_date)
            
        if self.send_email_bool:
            self.send_email('Final datasets are built and pruned!', start_time)
    
    def get_files(self, file_type):
        """Returns a list of all file names of the given file type."""
        paths = []
        for d in tqdm(self.dates, desc='Getting all files paths for each date:'):
            path = join(self._BASE_PATH,d)
            if self.times is None:
                for (dir_path, _, file_names) in os.walk(path):
                    file_names = [join(dir_path, f) for f in file_names if file_type in f]
                    paths.extend(file_names) 
            else:
                # For debugging the system. 
                for t in self.times:
                    dtry = join(path,t)
                    file_names = [join(dtry,f) for f in os.listdir(dtry) if file_type in f]
                    paths.extend(file_names) 
        
        paths.sort()
        
        return paths 
    
    def files_to_run(self, original_type='30M', new_type = 'ENSEMBLETRACKS'):
        """ Checks that files are created! """
        files_to_create = [] 
        
        files = self.get_files(original_type)
        
        if new_type in ['MLDATA', 'MLTARGETS']:
            new_files = [f.replace(original_type, new_type).replace('.nc', '.feather') for f in files]
        else:
            new_files = [f.replace(original_type, new_type) for f in files]
        
        # Check if "new" files already exist.
        all_exist = [exists(f) for f in new_files]
        if not all(all_exist):
            inds = [i for i, x in enumerate(all_exist) if not x]
            for i in tqdm(inds, desc='Getting files to run:'):
                ##print('debug', f"{new_files[i]} does not exist, but {files[i]} does!") 
                files_to_create.append(files[i]) 
        
        files_to_create.sort()
        
        return files_to_create
    
    def get_files_for_ml(self):
        """ Get the summary files for the ML feature extraction """
        # Get the summary file path directories.
        track_files = self.files_to_run('ENSEMBLETRACKS', 'MLDATA')
        
        delta_time_step = int(self._DURATION / self._DT)
        
        paths = [] 
        for track_file in tqdm(track_files, desc='Getting files for ML:'):
            indir = Path(track_file).parent.resolve()
            ti = int(decompose_file_path(track_file)['TIME_INDEX'])
            try:
                env_file = glob(join(indir, f'wofs_ENV_{ti-delta_time_step:02d}*'))[0]
                svr_file = env_file.replace('ENV', 'SVR')

                ens_files = [glob(join(indir, f'wofs_ENS_{t:02d}*'))[0] for t in range(ti-delta_time_step, ti+1)]
                files = {'track_file' : track_file, 
                     'env_file'   : env_file,
                     'svr_file'   : svr_file, 
                     'ens_file'   : ens_files,
                }
                if self.keep_existing_ml_files:
                    if not exists(track_file.replace('ENSEMBLETRACKS', 'MLDATA').replace('.nc', '.feather')):
                        paths.append(files)
                else:
                    paths.append(files)
                    
            except:
                print(f'Issue with {track_file} for ml files')
        
        return paths 
  