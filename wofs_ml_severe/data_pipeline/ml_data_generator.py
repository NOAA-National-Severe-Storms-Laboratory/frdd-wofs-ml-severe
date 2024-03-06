#======================================================
# Generates the ML dataset from the WoFS summary files
# and ensemble storm tracks. This script is used in 
# real-time to generate the data from the WoFS-ML-Severe 
# products. 
# 
# Author: Montgomery Flora (Git username : monte-flora)
# Email : monte.flora@noaa.gov 
#======================================================

# Python Modules
from os.path import join, basename, dirname
import argparse
import itertools
import pathlib
import gc 
# Only model pickle files
import os, sys
sys.path.append(os.getcwd())
from skimage.measure import regionprops
import json 
import math
from glob import glob

# Third part modules
import numpy as np
import xarray as xr
import pandas as pd
import joblib

# WoFS modules 
_base_module_path = '/home/monte.flora/python_packages/frdd-wofs-post'
_base_mp_path = '/home/monte.flora/python_packages/MontePython'
#import sys
#sys.path.insert(0, _base_module_path)
#sys.path.insert(0,_base_mp_path)
import monte_python 

from ..common.multiprocessing_utils import run_parallel, to_iterator
from ..common.util import decompose_file_path, get_target_str, is_list, get_time_str, fix_data

from wofs.common import remove_reserved_keys
from wofs.common.zarr import open_dataset, normalize_filename
from wofs.post.utils import (
    save_dataset,
    generate_track_filename,
    load_multiple_nc_files,
    load_yaml
)

# Personal Modules 
from ..io.load_ml_models import load_ml_model, load_calibration_model, load_ml_models_2024
from ..io.io import get_numeric_init_time
from .storm_based_feature_extracter import StormBasedFeatureExtracter
from .local_explainer import LocalExplainer


def simplify_target(target_str): 
    translate_dict = {'severe_mesh' : 'hail',
                      'severe_wind' : 'wind', 
                      'severe_torn' : 'tornado', 
                      'any_severe' : 'all_severe',
                      'any_sig_severe' : 'all_sig_severe'
                     }
    return translate_dict.get(target_str, target_str)


class MLDataGenerator: 
    """
    MLDataGenerator handles the data loading, feature extraction, and prediction of the 
    the WoFS-ML-Severe model suite. 
    
    Attributes
    -------------------
    TEMP : True/False
       If True, then current parts of the data pre-processing procedure are modified 
       to ensure the data is correct with the current renditions of the ML models.
       
       Current TEMPS: 
       
       * Converting the mid-level temps from deg C to deg F. In the 2023-current summary files, 
       the mid-level temps are in deg C. The existing local summary files, however, were 
       generated with deg F. 
       
    keyword args
       - paths : dict 
           dict with keys = ['track_file', 'env_file', 'ens_file', 'svr_file']
           'ens_file' should contain 6 file paths for the 30-min duration while the
           other keys will have a single file path
           
       - ml_config_path : path-like, str
           Path to the yaml configuration file for the ML.  
    """
    ### Monte: Removed old inputs 08/24/2022.
    ### Removed retro 18 March 2023 
    def __init__(self, TEMP=True, debug=False, outdir=None, 
                 explain=True, model_path=None, **kwargs):
        
        self.TEMP = TEMP
        
        self.debug = debug
        self._outdir = outdir
        
        self.ml_config_path = kwargs.get('ml_config_path', None)
        self.explain=explain
        self.model_path = model_path
    
    
    def _load_config(self, path_to_summary_file):
        """Loads the YAML config file for the ML dataset"""
        if self.ml_config_path is None:
            path = join(pathlib.Path(__file__).parent.parent.resolve(), 'conf')
            ml_config_path = join(path, 'default_ml_config.yml')
        else:
            ml_config_path = self.ml_config_path

        print(f'Loading config file: {ml_config_path}....')    
            
        return load_yaml(ml_config_path)
    

    def __call__(self,
                 paths, 
                 n_processors=1, 
                 realtime=True, 
                outdir=None, 
                append=False, old_file_format=True):
        
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
        
        self.predict  = predict
        self.explain  = explain
        self.realtime = realtime
        self.old_file_format = old_file_format
        
        func = self._append if append else self.generate_ml_severe_files
        
        if isinstance(paths, list):
            run_parallel(
            func=func,
            nprocs_to_use=n_processors,
            args_iterator=to_iterator(paths),
            description='Generating ML Data'
            )
        else:
            ### Josh: if we are only loading one set of files, we don't need the Pool spinup 
            return func(paths)
    
    def to_2d(self, predictions, forecast_objects, object_labels, shape_2d):
        """Convert column vector predictions to 2D"""
        forecasts_2d = np.zeros((shape_2d), dtype=np.float32)
        for i, label in enumerate(object_labels.values):
            forecasts_2d[forecast_objects==label] = predictions[i]

        return forecasts_2d

    def to_xarray(self, data, storm_objects, ds_subset, ensemble_track_file, explainability_files):
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
    
        if ensemble_track_file is None:
            return ds 
    
        save_nc_file = ensemble_track_file.replace('ENSEMBLETRACKS', 'MLPROB')
        ###print(f'Saving {save_nc_file}...')
        if self.debug:
            save_nc_file = join(self._outdir, basename(save_nc_file))
        
        save_dataset(save_nc_file, ds)

        return [save_nc_file] + explainability_files
        
    def is_there_an_object(self, storm_objects):   
        return np.max(storm_objects) > 0

    def get_predictions(self, time, dataframe, storm_objects, ds_subset, ml_config, ens_probs, ensemble_track_file=None):
        """
        Produces 2D probabilistic predictions from the ML model and baseline system
        """
        prediction_data = {}
        object_labels = dataframe['label']
        times = [time] if not is_list(time) else time
        
        explainability_files = [] 
        
        # Create the trimmed objects.
        tag='trimmed'
        qc_params = [('trim', (9/18, 4/18))]
        object_props = regionprops(storm_objects, ens_probs) 
        qcer = monte_python.QualityControler()
        storm_objects_trimmed, _ = qcer.quality_control(ens_probs, storm_objects, object_props, qc_params)
        
        ml_model_path = os.path.join(self.model_path, ml_config['MODEL_NAMES'][-1])
        _, _, features = load_ml_models_2024(ml_model_path, return_features=True)
        
        X = dataframe[features] 
        X = fix_data(X)
        # Deprecated due to issues with Initialization time. 
        # Get the numeric init time 
        if 'Initialization Time' in X.columns:
            X = get_numeric_init_time(X)
        
        for model_fname in ml_config['MODEL_NAMES']:
            model, target = load_ml_models_2024(os.path.join(self.model_path, model_fname))
            target = simplify_target(target)

            # Compute the probabilities or hail size.
            # TODO: refactor for the hail regression model.
            if 'Regressor' in model_fname:
                predictions = model.predict(X)
                # Temporary fix for negative hail sizes! 
                predictions[predictions<=0.0] = 0.0
                hail_size = predictions
            else:
                predictions = model.predict_proba(X)[:,1]
            
            # Create the full and trimmed tracks and save them to the dataset. 
            for name, objs in zip(['full', 'trimmed'], [storm_objects,storm_objects_trimmed]):
                predictions_2d = self.to_2d(predictions, objs, object_labels, shape_2d=storm_objects.shape)
                prediction_data[f'ML__{target}__{name}'] = (['NY', 'NX'], predictions_2d)
            
            # Add trimmed tracks to save files. 
            prediction_data[f'trimmed_tracks'] = (['NY', 'NX'], storm_objects_trimmed)
            
            # Generate the local explainability JSON. 
            if self.explain:
                explainfile = self.generate_explainability_json(model, X, target, dataframe, features, 
                                     ensemble_track_file, ml_config, hail_size=hail_size, scale='local'                               
                                )    
                explainability_files.append(explainfile) 
                
                # Generate the global explainability JSON. 
                global_explainfile = self.generate_explainability_json(model, 
                                                                              X, target, 
                                                                              dataframe, features, 
                                                                             ensemble_track_file, ml_config,
                                                                             hail_size=hail_size, 
                                                                             scale = 'global')    
                explainability_files.append(global_explainfile)     
                
                
        return self.to_xarray(prediction_data, storm_objects, ds_subset, ensemble_track_file, explainability_files)

    def generate_explainability_json(self, model, X, 
                                     target, dataframe, features, 
                                     ensemble_track_file, ml_config, hail_size=None, scale='local'
                                ): 
        """Generate the local or global explainability JSON file"""
        target_str = ml_config['TARGET_CONVERTER'].get(target, target)
        
        # Load the round_dict 
        json_file = join(pathlib.Path(__file__).parent.parent.resolve(), 'json', f'min_max_vals_{target_str}.json' )
        
        with open(json_file) as f:
            results = json.load(f)
    
        round_dict = {f : results[f]['round_int'] for f in features}
    
        # Get the metadata like the object coords. Also, adding additional data. 
        metadata = dataframe[['label', 'obj_centroid_x', 'obj_centroid_y', 'ens_track_prob']]
        metadata['ens_track_prob'] = (metadata['ens_track_prob']*18).astype(int)
        metadata['hail_size'] = np.round(hail_size, 2) 

        # Round the data. 
        dataframe = dataframe.round(round_dict)
        X = X.round(round_dict) 
        
        # Save subset of data for the explainability graphics. 
        explain_fname = ensemble_track_file.replace('ENSEMBLETRACKS', 
                                                    f'{scale.upper()}EXPLAIN__{target_str}').replace('.nc', '.json')
        
        if scale == 'local':      
            # The local explainability uses the input (val*coef) for the logistic regression model. 
            # It sums together attributions of the log-odds for a feature parent (i.e., all variations on 
            # mid-level UH). or use the SHAP method to determine the top features for a particular example. 
            X_train = pd.read_feather(os.path.join(ml_config['ML_MODEL_PATH'], 'shap_samples.feather'))
            X_train = X_train[X.columns]
            
            explainer = LocalExplainer(model, X, X_train=X_train)
            top_features, top_values = explainer.top_features(target_str, method='shap')
            
        else:
            # The global explainability uses a static list of top features, which was determined 
            # by the features with the highest sum total of abs(coefs) for the first and second hour datasets. 
            n_examples = len(dataframe)
            top_features_ = list(ml_config['TOP_FEATURES'])
            top_values = dataframe[top_features_].values 
            top_features = [list(top_features_) for _ in range(n_examples)]
            
    
        val_df = pd.DataFrame(top_values, columns=[f'Feature Val {i+1}' for i in range(5)])
        feature_df = pd.DataFrame(top_features, columns=[f'Feature Name {i+1}' for i in range(5)])
    
        total_df = pd.concat([val_df, feature_df, metadata], axis=1)

        print(f'Saving {explain_fname}...')
        total_df.to_json(explain_fname)
    
        return explain_fname

    def decompose_path(self, path):
        """Get the Run Date and Initialization time from the file path"""
        outer_path = dirname(path)
        init_time = basename(outer_path)
        run_date = basename(dirname(outer_path))
    
        return run_date, init_time
    
    
    def _append(self, file_dict):
        """
        This function is used to append columns onto existing ML dataframes.
        This is to prevent re-generating the full dataset when adding new 
        predictors. Therefore, this function is flexible and may change in 
        future. 
        """
        ensemble_track_file = file_dict['track_file']
        
        base_path = dirname(ensemble_track_file) 
        t = int(decompose_file_path(ensemble_track_file)['TIME_INDEX'])
        ens_files = [glob(os.path.join(base_path, f'wofs_ENS_{_t:02d}*'))[0] for _t in range(t-6, t+1)]
        
        svr_file = ens_files[0].replace('ENS', 'SVR') 
        
        ml_config = self._load_config(ensemble_track_file) 
        
        # See if there are tracks
        try:
            ensemble_track_ds = open_dataset(ensemble_track_file, decode_times=False)
        except OSError:
            print(f'Unable to load {ensemble_track_file}')
            return None
            
        storm_objects = ensemble_track_ds['w_up__ensemble_tracks'].values
        intensity_img = ensemble_track_ds['w_up__ensemble_probabilities'].values
        updraft_tracks = ensemble_track_ds['updraft_tracks'].values

        extracter = StormBasedFeatureExtracter(ml_config, cond_var=None)
        
        # See if there are tracks
        if self.is_there_an_object(storm_objects):
            coord_vars = ["xlat", "xlon", "hgt"]
            object_props_df = extracter.get_object_properties(storm_objects, intensity_img)
            labels = object_props_df['label']
            
            # Extract the spatial-based features.
            # Compute ens. statistics (data is still 2d at this point). 
            
            # Load the SVR file
            ds_svr = open_dataset(svr_file, decode_times=False)
            try:
                svr_data = {var: ds_svr[var].values for var in ['cape_sfc', 
                                                                'cin_sfc', 
                                                                'cape_mu', 
                                                                'cin_mu', 
                                                                'lcl_sfc', 
                                                                'lcl_mu', 
                                                               ]}
            except:
                print(f"Issue with {svr_file}. Likely due to the missing SRH0to1")
                ds_env.close()
                gc.collect()
                return None 
            
            ensemble_data = extracter._compute_ens_stats(svr_data)
            
            df_spatial = extracter.extract_spatial_features_from_object( 
                ensemble_data, 
                storm_objects, 
                labels
            )

            # Open existing dataframe
            ml_data_path = ensemble_track_file.replace('ENSEMBLETRACKS', 'MLDATA').replace('.nc', '.feather')
            ml_df = pd.read_feather(ml_data_path)
            
            new_df = pd.concat([ml_df, df_spatial], axis=1)
            new_df.reset_index(inplace=True, drop=True) 
        
            # overwrite the existing file. 
            
            new_df.to_feather(ml_data_path)
            ensemble_track_ds.close()
            
        return ml_data_path 
        
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
        try:
            ensemble_track_ds = open_dataset(ensemble_track_file, decode_times=False)
        except OSError:
            print(f'Unable to load {ensemble_track_file}')
            return None
            
        storm_objects = ensemble_track_ds['w_up__ensemble_tracks'].values
        intensity_img = ensemble_track_ds['w_up__ensemble_probabilities'].values
        updraft_tracks = ensemble_track_ds['updraft_tracks'].values
        generated_files = []
        

        if self.is_there_an_object(storm_objects):
            # Load ENV file
            ds_env = open_dataset(env_file, decode_times=False)
            ds_subset = ds_env[['xlat', 'xlon', 'hgt']]
            try:
                env_data = {var: ds_env[var].values for var in ml_config['ENV_VARS']}
            except:
                print(f"Issue with {env_file}. Likely due to the missing variable for eariler WoFS years!")
                ds_env.close()
                gc.collect()
                return None
                
            if self.TEMP: 
                # Deprecated on 17 March 2023 
                # Convert lapse rate from C/KM back to C 
                #env_data['mid_level_lapse_rate'] = env_data['mid_level_lapse_rate']*2.67765
                #env_data['low_level_lapse_rate'] = env_data['low_level_lapse_rate']*3.0  
                
                # Convert from deg C to deg F 
                temp_vars = ['temperature_850', 
                             'temperature_700',
                             'temperature_500', 
                             'td_850',
                             'td_700',
                             'td_500', 
                            ]
                
                for var in temp_vars:
                    # C -> F 
                    if var in env_data.keys():
                        env_data[var] = (1.8 * env_data[var]) + 32.  
                
            # Some environmental variables may be in the ENS files
            if len(ml_config['ENV_IN_ENS_VARS']) > 0:
                ds_ens = open_dataset(ens_files[0], decode_times=False)
                ens_data = {var: ds_ens[var].values for var in ml_config['ENV_IN_ENS_VARS']}
                env_data = {**env_data, **ens_data}
                ds_ens.close()
                del ds_ens

            # Load the SVR file
            ds_svr = open_dataset(svr_file, decode_times=False)
            try:
                svr_data = {var: ds_svr[var].values for var in ml_config['SVR_VARS']}
            except:
                print(f"Issue with {svr_file}. Likely due to the missing SRH0to1")
                ds_env.close()
                gc.collect()
                return None 
                
            coord_vars = ["xlat", "xlon", "hgt"]
            try:
                multiple_datasets_dict, coord_vars_dict, dataset_attrs, var_attrs  = load_multiple_nc_files(
                        ens_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENS_VARS'])
            except: 
                gc.collect()
                print(f"Issue with {ens_files}. Likely a missing variable ('uh_0to2')")
                return None 
       
            # Convert data to xarray.Dataset 
            storm_ds = xr.Dataset(multiple_datasets_dict)
            
            # Initialize the Extracter class. 
            extracter = StormBasedFeatureExtracter(ml_config, cond_var=None)

            # Combine the ENV and SVR file output together. 
            env_data = {**env_data, **svr_data}
        
            # Determine the Run date and Init. time from the file path. 
            run_date, init_time = self.decompose_path(env_file) 
            
            # Run the extracter. Returns the data as a dataframe. 
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

            # Add the run date as metadata. 
            dataframe['Run Date'] = [int(run_date)] * len(dataframe)

            # Add the forecast time index. 
            time_index = decompose_file_path(env_file)['TIME_INDEX']
            dataframe['forecast_time_index'] = [int(time_index)] * len(dataframe)
            
            # If we are running in realtime, then we want to generate the predictions.
            if self.predict:
                # Check the time index to determine the right model. 
                time_index = int(env_file.split('_')[-4])
                time = get_time_str(time_index)
                
                mlprob_file = self.get_predictions(time, dataframe, storm_objects, ds_subset, ml_config, 
                                                   intensity_img, ensemble_track_file)
                generated_files.extend(mlprob_file)
            
            ensemble_track_ds.close()
            del ensemble_track_ds
        
            # Save the dataframe to a JSON file. 
            save_df_file = ensemble_track_file.replace('ENSEMBLETRACKS', 
                                                       'MLDATA').replace('.nc', '.feather').replace('.json', '.feather')
            
            if self.debug:
                save_df_file = join(self._outdir, basename(save_df_file))
            
            dataframe.to_feather(save_df_file)
        
        
            # Generate the EXPLAIN json file (soon to be deprecated!).
            if self.explain:
                explain_fname = ensemble_track_file.replace('ENSEMBLETRACKS', 'EXPLAIN').replace('.nc', '.json') 
                # Get the feature values for the EXPLAIN file. 
                df_subset = dataframe[list(ml_config['FEATURE_SUBSET_DICT'].keys())+\
                                  ['label', 'obj_centroid_x', 'obj_centroid_y']] 
            
                df_subset = df_subset.rename(columns = ml_config['FEATURE_SUBSET_DICT']) 
                # Round the values. 
                df_subset = df_subset.round(ml_config['ROUNDING'])
                
                # Temporary conversion!!
                df_subset['0-3km_lapse_rate' ] *= -1
                df_subset['500-700mb_lapse_rate' ] *= -1
                
                # Save the EXPLAIN file. 
                df_subset.to_json(explain_fname)

            return [save_df_file, explain_fname] + generated_files

        else:
            if self.predict:
                ds_env = open_dataset(env_file, decode_times=False)
                ds_subset = ds_env[['xlat', 'xlon', 'hgt']]
                prediction_data={}
                for pair in itertools.product(ml_config['MODEL_NAMES'], ml_config['TARGETS']):
                    model_name, target = pair
                    model_name_str = 'ML' if model_name == 'Average' else model_name
                    target_str = get_target_str(target)
                    
                    for name in ['full', 'trimmed']:
                        prediction_data[f'{model_name_str}__{target_str}__{name}'] = (['NY', 'NX'],
                                                          np.zeros((storm_objects.shape), dtype=np.int32))
                    # Add trimmed tracks to save files. 
                    prediction_data[f'trimmed_tracks'] = (['NY', 'NX'], np.zeros((storm_objects.shape), dtype=np.int32))
                    
                generated_files.extend(self.to_xarray(prediction_data, storm_objects, ds_subset, ensemble_track_file, []))
                ds_env.close()
                gc.collect()
                del ds_env
        
        gc.collect()
        
        return generated_files
    

    
