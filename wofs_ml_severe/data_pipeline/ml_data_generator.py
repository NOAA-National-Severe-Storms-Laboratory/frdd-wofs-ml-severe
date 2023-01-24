#======================================================
# Generates the ML dataset from the WoFS summary files
# and ensemble storm tracks.
# 
# Author: Montgomery Flora (Git username : monte-flora)
# Email : monte.flora@noaa.gov 
#======================================================

# Python Modules
from os.path import join, basename, dirname
import argparse
import itertools
import pathlib
# Only model pickle files
import os, sys
sys.path.append(os.getcwd())

# Third part modules
import numpy as np
import xarray as xr
import pandas as pd

# WoFS modules 
_base_module_path = '/home/monte.flora/python_packages/master/WoF_post'
import sys
sys.path.append(_base_module_path)

from ..common.multiprocessing_utils import run_parallel, to_iterator
from ..common.util import decompose_file_path

from wofs.common import remove_reserved_keys
from wofs.common.zarr import open_dataset, normalize_filename
from wofs.post.utils import (
    save_dataset,
    generate_track_filename,
    load_multiple_nc_files,
    load_yaml
)

# Personal Modules 
from ..io.load_ml_models import load_ml_model, load_calibration_model
from .storm_based_feature_extracter import StormBasedFeatureExtracter
# Temporary until the ML models are re-trained 
# with the new naming convention!!!
from .name_mapper import name_mapping


class MLDataGenerator: 
    """
    MLDataGenerator handles the data loading, feature extraction, and prediction of the 
    the WoFS-ML-Severe model suite. 
    
    Attributes
    -------------------
    TEMP : True/False
       If True, the previous ML configuration and naming convention from the 2021 paper is used!
       
    retro : True/False (default=False) 
        If True, data is pulled from /work rather than /scratch
     
    keyword args
       - paths : dict 
           dict with keys = ['track_file', 'env_file', 'ens_file', 'svr_file']
           'ens_file' should contain 6 file paths for the 30-min duration while the
           other keys will have a single file path
           
       - ml_config_path : path-like, str
           Path to the yaml configuration file for the ML.  
    """
    ### Monte: Removed old inputs 08/24/2022.
    def __init__(self, TEMP=True, retro=False, debug=False, **kwargs):
        
        self.TEMP = TEMP
        self.retro=retro
        self.debug = debug
        
        self.ml_config_path = kwargs.get('ml_config_path', None)

    def __call__(self,
                 paths, 
                 n_processors=1, 
                 realtime=True ):
        """Runs the data generator
        
        Parameters
        -------------------
        paths  : dict or list of dicts with file paths
            If dict, contains the following keys: 
                'track_file' : ENSEMBLETRACK Summary file 
                env_file, svr_file, and env_file : ENV, SVR, and ENS (6) files

            files can also be a list of dicts of the type above for 
            parallel processing of multiple cases. 

        n_processors : int (default=1)
            The number of processors for multiprocessing
            By default, we use 1 processor for the real-time code
        
        realtime : True/False
            If True, the generator will produce the data, predictions, and explainability data. 
            else only the data is produced.  
        
        Returns
        -------------------
        Saves the MLDATA feather file. 
        """
        runtype = 'rlt' if realtime else 'rto' 
        
        predict=False if runtype == 'rto' else True
        explain=False if runtype == 'rto' else True
        
        self.predict=predict
        self.explain = explain
        self.realtime = realtime
        
        if isinstance(paths, list):
            run_parallel(
            func=self.generate_ml_severe_files,
            nprocs_to_use=n_processors,
            args_iterator=to_iterator(paths),
            description='Generating ML Data'
            )
        else:
            ### Josh: if we are only loading one set of files, we don't need the Pool spinup 
            return self.generate_ml_severe_files(paths)
    
    def to_2d(self, predictions, forecast_objects, object_labels, shape_2d):
        """Convert column vector predictions to 2D"""
        forecasts_2d = np.zeros((shape_2d), dtype=np.float32)
        for i, label in enumerate(object_labels.values):
            forecasts_2d[forecast_objects==label] = predictions[i]

        return forecasts_2d

    def to_xarray(self, data, storm_objects, ds_subset, ensemble_track_file):
        """
        Convert the 2D prediction fields to xarray dataset. 
        """
        ds = xr.Dataset(data)
        ds['ensemble_tracks'] = (['NY', 'NX'], storm_objects)
    
        # Adding xlat, xlong, and hgt and attributes of ds_env
        ds = xr.merge([ds, ds_subset], combine_attrs="no_conflicts", compat="override") 

        # cleanup reserved netcdf keys
        for var in ds.data_vars:
            ds.data_vars[var].attrs = remove_reserved_keys(ds.data_vars[var].attrs)
    
        ds.attrs = remove_reserved_keys(ds.attrs)
    
        save_nc_file = ensemble_track_file.replace('ENSEMBLETRACKS', 'MLPROB')
        ###print(f'Saving {save_nc_file}...')
        save_dataset(save_nc_file, ds)

        return save_nc_file
        
    def is_there_an_object(self, storm_objects):   
        return np.max(storm_objects) > 0

    def get_predictions(self, time, dataframe, storm_objects, ds_subset, ensemble_track_file):
        """
        Produces 2D probabilistic predictions from the ML model and baseline system
        """
        prediction_data = {}
        object_labels = dataframe['label']
        for pair in itertools.product(ml_config['MODEL_NAMES'], ml_config['TARGETS']):
            model_name, target = pair
            parameters = {
                'time' : time,
                'target' : target,
                'drop_opt' : '',
                'model_name' : model_name,
                'ml_config' : ml_config,
            }
            if model_name != 'Baseline':
                model_dict = load_ml_model(**parameters)
                model = model_dict['model']
                features = model_dict['features']
            
                if self.TEMP:
                    corrected_feature_names = name_mapping(features)
                    new_features = [corrected_feature_names.get(f,f) for f in features]
                    X = dataframe[new_features]
                else:
                    X = dataframe[features]
                
                predictions = model.predict_proba(X)[:,1]
            else:
                iso_reg = load_calibration_model(**parameters)
                raw_predictions = dataframe[ml_config['BASELINE_VARS'][target]].values
                predictions = iso_reg.predict(raw_predictions)
        
            predictions_2d = self.to_2d(predictions, storm_objects, object_labels, shape_2d=storm_objects.shape)
            prediction_data[f'{model_name}__{target}'] = (['NY', 'NX'], predictions_2d)
    
        return self.to_xarray(prediction_data, storm_objects, ds_subset, ensemble_track_file)

    
    def _correct_naming(self, data):
        """ Remove the '_instant' for the intra-storm variables like UH"""
        varnames = list(data.keys())
        for v in varnames:
            if 'instant' in v:
                data[v.split('_instant')[0]] = data[v]
                del data[v]
    
        return data 
    
    def _load_config(self, path_to_summary_file):
        """Loads the YAML config file for the ML dataset"""
        path = join(pathlib.Path(__file__).parent.parent.resolve(), 'conf')
        if self.ml_config_path is not None:
              self.ml_config_path = ml_config_path
        else:
            if self.realtime: 
                ml_config_path = join(path, 'ml_config_realtime.yml')
            else:
                comps = decompose_file_path(path_to_summary_file)
                year = comps['VALID_DATE'][:4]
                if year == '2017':
                    ml_config_path = join(path, 'ml_config_2017.yml')
                elif year in ['2018', '2019']:
                    ml_config_path = join(path, 'ml_config_2018-19.yml')
                else:
                    ml_config_path = join(path, 'ml_config_2020-current.yml')
        
        ###print(f'{ml_config_path=}')
        return load_yaml(ml_config_path)
    
    def decompose_path(self, path):
        """Get the Run Date and Initialization time from the file path"""
        outer_path = dirname(path)
        init_time = basename(outer_path)
        run_date = basename(dirname(outer_path))
    
        return run_date, init_time
    
    def generate_ml_severe_files(self, file_dict):
        """
        Generates the dataframe of input features, the file for the explainability graphic, 
        and the 2D ML predictions used for webviewer graphics. 
        """    
        ensemble_track_file = file_dict['track_file'] 
        env_file= file_dict['env_file'] 
        svr_file= file_dict['svr_file'] 
        ens_files= file_dict['ens_file']
        ml_config = self._load_config(ensemble_track_file) 
              
        ########
        if self.debug:
            print('REMEMBER THAT DEBUG IS ON!!!!!')
        ########
        
        # See if there are tracks
        ensemble_track_ds = open_dataset(ensemble_track_file, decode_times=False)
        storm_objects = ensemble_track_ds['w_up__ensemble_tracks'].values
        intensity_img = ensemble_track_ds['w_up__ensemble_probabilities'].values
        updraft_tracks = ensemble_track_ds['updraft_tracks'].values
        generated_files = []

        if self.is_there_an_object(storm_objects):
            # Load ENV file
            ds_env = open_dataset(env_file, decode_times=False)
            ds_subset = ds_env[['xlat', 'xlon', 'hgt']]
            env_data = {var: ds_env[var].values for var in ml_config['ENV_VARS']}

            if self.TEMP: 
                # Convert lapse rate from C/KM back to C 
                env_data['mid_level_lapse_rate'] = env_data['mid_level_lapse_rate']*2.67765
                env_data['low_level_lapse_rate'] = env_data['low_level_lapse_rate']*3.0  
                
                # Convert from deg F to deg C 
                temp_vars = ['temperature_850','temperature_700',
                             'temperature_500','td_850','td_700','td_500']
                for var in temp_vars:
                    env_data[var] = (5./9.) * (env_data[var] - 32.)
                
            # Some environmental variables may be in the ENS files
            if len(ml_config['ENV_IN_ENS_VARS']) > 0:
                ds_ens = open_dataset(ens_files[0], decode_times=False)
                ens_data = {var: ds_ens[var].values for var in ml_config['ENV_IN_ENS_VARS']}
                env_data = {**env_data, **ens_data}
                ds_ens.close()
                del ds_ens

            # Load the SVR file
            ds_svr = open_dataset(svr_file, decode_times=False)
            svr_data = {var: ds_svr[var].values for var in ml_config['SVR_VARS']}

            coord_vars = ["xlat", "xlon", "hgt"]
            try:
                multiple_datasets_dict, coord_vars_dict, dataset_attrs, var_attrs  = load_multiple_nc_files(
                        ens_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENS_VARS'])
            except KeyError:
                print(f"Issue: {ens_files[0]}, {ml_config['ENS_VARS']}")
                print('Manually replacing uh_0to2...')
                if 'uh_0to2' in ml_config['ENS_VARS']:
                    load_vars=ml_config['ENS_VARS']
                    load_vars.remove('uh_0to2')
                    load_vars.append('uh_0to2_instant')
                    multiple_datasets_dict, coord_vars_dict, dataset_attrs, var_attrs  = load_multiple_nc_files(
                        ens_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENS_VARS']) 
                
                
            # Correct the naming convention. 
            multiple_datasets_dict = self._correct_naming(multiple_datasets_dict)
           
            storm_ds = xr.Dataset(multiple_datasets_dict)
            
            if self.TEMP:
                # CTT is in the ENS file. Converting to deg C. 
                storm_ds['ctt'] = (5./9.) * (storm_ds['ctt'] - 32.)
            
            extracter = StormBasedFeatureExtracter(ml_config, TEMP=self.TEMP)
        
            env_data = {**env_data, **svr_data}
        
            run_date, init_time = self.decompse_path(env_file) 
        
            dataframe = extracter.extract(storm_objects,
                                      intensity_img,
                                      storm_ds, 
                                       env_data, 
                                       init_time=str(init_time),
                                        updraft_tracks=updraft_tracks,
                                     ) 

            # Close the netcdf files
            storm_ds.close()
            ds_env.close()
            ds_svr.close()
            del storm_ds, ds_env, ds_svr

            # Add the run date
            dataframe['Run Date'] = [int(run_date)] * len(dataframe)

            if self.predict:
                # Check the time index to determine whether it is valid for 0-1 or 1-2 hr. 
                time = 'first_hour' if int(env_file.split('_')[-4]) <= 20 else 'second_hour'
                mlprob = self.get_predictions(time, dataframe, storm_objects, ds_subset, ensemble_track_file)
                generated_files.append(mlprob)
            
            ensemble_track_ds.close()
            del ensemble_track_ds
        
            # Save the dataframe to a JSON file. 
            save_df_file = ensemble_track_file.replace('ENSEMBLETRACKS', 
                                                       'MLDATA').replace('.nc', '.feather').replace('.json', '.feather')
            
            if self.debug:
                save_df_file = basename(save_df_file)
            
            ###print(f'Saving the dataframe @ {save_df_file}...')
            dataframe.to_feather(save_df_file)
        
            # Save subset of data for the explainability graphics. 
            subset_fname = ensemble_track_file.replace('ENSEMBLETRACKS', 'EXPLAIN').replace('.nc', '.json') 
            
            if self.explain:
                df_subset = dataframe[list(ml_config['FEATURE_SUBSET_DICT'].keys())+\
                                  ['label', 'obj_centroid_x', 'obj_centroid_y']] 
            
                df_subset = df_subset.rename(columns = ml_config['FEATURE_SUBSET_DICT']) 
        
                if self.TEMP:
                    # Convert the lapse rates back into lapse rates rather than temp diffs. 
                    df_subset['0-3km_lapse_rate']/=-3.0
                    df_subset['500-700mb_lapse_rate']/=-2.67765
            
                # Round the values. 
                df_subset = df_subset.round(ml_config['ROUNDING'])
            
                ###print(f'Saving the explainability dataset @ {subset_fname}...')
                if self.debug:
                    subset_fname = basename(subset_fname)
                df_subset.to_json(subset_fname)
        
            return [ save_df_file, subset_fname ] + generated_files

        else:
            if self.predict:
                ds_env = open_dataset(env_file, decode_times=False)
                ds_subset = ds_env[['xlat', 'xlon', 'hgt']]
                prediction_data={}
                for pair in itertools.product(ml_config['MODEL_NAMES'], ml_config['TARGETS']):
                    model_name, target = pair
                    prediction_data[f'{model_name}__{target}'] = (['NY', 'NX'],
                                                          np.zeros((storm_objects.shape), dtype=np.int32))
                generated_files.append(self.to_xarray(prediction_data, storm_objects, ds_subset, ensemble_track_file))
                ds_env.close()
                del ds_env
                
        return generated_files
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str)
    parser.add_argument("-n", "--n_processors", type=float)
    parser.add_argument("--dt", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--runtype", type=str)
    parser.add_argument("--config", type=str)

    args = parser.parse_args()  
    
    mlops = MLOps(indir=args.indir, 
                  dt=args.dt, 
                  nt=args.nt, 
                  ml_config_path=args.config) 
    mlops(args.n_processors, args.runtype)
    
