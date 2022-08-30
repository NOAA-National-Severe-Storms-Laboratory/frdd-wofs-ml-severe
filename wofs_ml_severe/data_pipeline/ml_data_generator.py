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

from wofs.common import remove_reserved_keys
from wofs.common.zarr import open_dataset, normalize_filename
from wofs.post.multiprocessing_script import run_parallel_realtime, to_iterator
from wofs.post.utils import (
    save_dataset,
    generate_track_filename,
    load_multiple_nc_files,
    generate_summary_file_name,
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
       - files_to_load : dict 
           dict with keys = ['track_file', 'env_file', 'ens_file', 'svr_file']
           'ens_file' should contain 6 file paths for the 30-min duration while the
           other keys will have a single file path
           
       - ml_config_path : path-like, str
           Path to the yaml configuration file for the ML.  
    """
    ### Monte: Removed old inputs 08/24/2022.
    def __init__(self, files_to_load, TEMP=True, retro=False, debug=False,**kwargs):
        
        self.TEMP = TEMP
        self.retro=retro
        self.files_to_load = files_to_load #kwargs.get('files_to_load', [])
        self.debug = debug

        # Deprecated
        #if len(self.files_to_load) == 0:
        #    self.files_to_load = self.get_files(indir, dt, nt,)
        
        self.ml_config_path = kwargs.get('ml_config_path', 
                                         join( pathlib.Path(__file__).parent.resolve(), 
                                              'ml_config_realtime.yml' ))

    def __call__(self, 
                 n_processors=1, 
                 runtype='rlt' ):
        """Runs the data generator
        
        Parameters
        -------------------
        n_processors : int (default=1)
            The number of processors for multiprocessing
            By default, we use 1 processor for the real-time code
        runtype : 'rto' or 'rlt'
            If 'rto', the generator will only produce the data. 
            If 'rlt', the generator will produce predictions 
            and the explainability data. 
        
        Returns
        -------------------
        Saves the MLDATA feather file. 
        """
        
        predict=False if runtype == 'rto' else True
        explain=False if runtype == 'rto' else True
        
        self.predict=predict
        self.explain = explain
        self.ml_config = load_yaml(self.ml_config_path)
        
        if len(self.files_to_load) > 1:
            run_parallel_realtime(
            func=self._worker,
            nprocs_to_use=n_processors,
            iterator=to_iterator(self.files_to_load),
            rtype=runtype,)
        ### Josh: if we are only loading one set of files, we don't need the Pool spinup  
        else:
            return self._worker(self.files_to_load[0])
    
    
    def _worker(self, kwargs):
        """Worker function for multiprocessing."""
        return self.generate_ml_severe_files(ensemble_track_file = kwargs['track_file'], 
                            env_file= kwargs['env_file'], 
                            svr_file= kwargs['svr_file'], 
                            ens_files= kwargs['ens_file'], 
                           )
    '''
    def get_files(self, indir, dt, nt, duration=30):
        """Get the prerequiste summary file paths"""
   
        delta_time_step = int(duration / dt)
        total_idx = (nt + delta_time_step) + 1

        summary_file_dir = indir
        summary_ens_files = generate_summary_file_name(
            indir=indir,
            outdir=indir,
            time_idxs=range(total_idx),
            mode="ENS",
            output_timestep=dt,
            first_hour=14,
        )

        iterator = range(nt+1)
        iterator_size = len(list(iterator))

        diry = 'work' if self.retro else 'scratch'
    
        files_to_load = []
        for i in iterator:
            # Add the ENS files
            ens_files = [
                join(summary_file_dir, f).replace('work',diry).replace('_ML','_RLT') 
                for f in summary_ens_files[i : i + delta_time_step + 1]
            ]
            fname = generate_track_filename(
                ens_files[0], ens_files[-1], duration=duration, nt = nt 
            )
        
            track_file = join(summary_file_dir, fname.replace('30M', 'ENSEMBLETRACKS'))
            env_file = ens_files[0].replace('ENS', 'ENV').replace('_ML','_RLT')
            svr_file = ens_files[0].replace('ENS', 'SVR').replace('_ML','_RLT')

            files_to_load.append( {'track_file': track_file, 
                         'ens_file' : ens_files, 
                         'env_file'  : env_file, 
                         'svr_file'  : svr_file,      
                        }) 
    
        return files_to_load 
    '''
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
        print(f'Saving {save_nc_file}...')
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
        for pair in itertools.product(self.ml_config['MODEL_NAMES'], self.ml_config['TARGETS']):
            model_name, target = pair
            parameters = {
                'time' : time,
                'target' : target,
                'drop_opt' : '',
                'model_name' : model_name,
                'ml_config' : self.ml_config,
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
                raw_predictions = dataframe[self.ml_config['BASELINE_VARS'][target]].values
                predictions = iso_reg.predict(raw_predictions)
        
            predictions_2d = self.to_2d(predictions, storm_objects, object_labels, shape_2d=storm_objects.shape)
            prediction_data[f'{model_name}__{target}'] = (['NY', 'NX'], predictions_2d)
    
        return self.to_xarray(prediction_data, storm_objects, ds_subset, ensemble_track_file)


    def generate_ml_severe_files(self, ensemble_track_file, env_file, svr_file, ens_files,):
        """
        Generates the dataframe of input features, the file for the explainability graphic, 
        and the 2D ML predictions used for webviewer graphics. 
        """
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
            env_data = {var: ds_env[var].values for var in self.ml_config['ENV_VARS']}

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
            if len(self.ml_config['ENV_IN_ENS_VARS']) > 0:
                ds_ens = open_dataset(ens_files[0], decode_times=False)
                ens_data = {var: ds_ens[var].values for var in self.ml_config['ENV_IN_ENS_VARS']}
                env_data = {**env_data, **ens_data}
                ds_ens.close()
                del ds_ens

            # Load the SVR file
            ds_svr = open_dataset(svr_file, decode_times=False)
            svr_data = {var: ds_svr[var].values for var in self.ml_config['SVR_VARS']}

            coord_vars = ["xlat", "xlon", "hgt"]
            multiple_datasets_dict, coord_vars_dict, dataset_attrs, var_attrs  = load_multiple_nc_files(
                ens_files, concat_dim="time", coord_vars=coord_vars,  load_vars=self.ml_config['ENS_VARS'])

            if self.TEMP:
                 # Convert naming of updraft helicity and vertical vorticity 
                multiple_datasets_dict['uh_2to5'] = multiple_datasets_dict['uh_2to5_instant']
                multiple_datasets_dict['uh_0to2'] = multiple_datasets_dict['uh_0to2_instant']
                multiple_datasets_dict['wz_0to2'] = multiple_datasets_dict['wz_0to2_instant']

                del multiple_datasets_dict['uh_2to5_instant']
                del multiple_datasets_dict['uh_0to2_instant']
                del multiple_datasets_dict['wz_0to2_instant']
            
            storm_ds = xr.Dataset(multiple_datasets_dict)
            
            if self.TEMP:
                # CTT is in the ENS file. Converting to deg C. 
                storm_ds['ctt'] = (5./9.) * (storm_ds['ctt'] - 32.)
            
            extracter = StormBasedFeatureExtracter(self.ml_config, TEMP=self.TEMP)
        
            env_data = {**env_data, **svr_data}
        
            dataframe = extracter.extract(storm_objects,
                                      intensity_img,
                                      storm_ds, 
                                       env_data, 
                                       init_time=str(env_file.split('_')[-2]),
                                        updraft_tracks=updraft_tracks,
                                     ) 

            # Close the netcdf files
            storm_ds.close()
            ds_env.close()
            ds_svr.close()
            del storm_ds, ds_env, ds_svr

            # Add the run date
            dataframe['Run Date'] = [int(env_file.split('_')[-3])] * len(dataframe)

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
            
            print(f'Saving the dataframe @ {save_df_file}...')
            dataframe.to_feather(save_df_file)
        
            # Save subset of data for the explainability graphics. 
            subset_fname = ensemble_track_file.replace('ENSEMBLETRACKS', 'EXPLAIN').replace('.nc', '.json') 
            
            if self.explain:
                df_subset = dataframe[list(self.ml_config['FEATURE_SUBSET_DICT'].keys())+\
                                  ['label', 'obj_centroid_x', 'obj_centroid_y']] 
            
                df_subset = df_subset.rename(columns = self.ml_config['FEATURE_SUBSET_DICT']) 
        
                if self.TEMP:
                    # Convert the lapse rates back into lapse rates rather than temp diffs. 
                    df_subset['0-3km_lapse_rate']/=-3.0
                    df_subset['500-700mb_lapse_rate']/=-2.67765
            
                # Round the values. 
                df_subset = df_subset.round(self.ml_config['ROUNDING'])
            
                print(f'Saving the explainability dataset @ {subset_fname}...')
                if self.debug:
                    subset_fname = basename(subset_fname)
                df_subset.to_json(subset_fname)
        
            return [ save_df_file, subset_fname ] + generated_files

        else:
            if self.predict:
                ds_env = open_dataset(env_file, decode_times=False)
                ds_subset = ds_env[['xlat', 'xlon', 'hgt']]
                prediction_data={}
                for pair in itertools.product(self.ml_config['MODEL_NAMES'], self.ml_config['TARGETS']):
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
    
