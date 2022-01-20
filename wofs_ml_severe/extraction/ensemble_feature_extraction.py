import numpy as np
import itertools
import xarray as xr 
from os.path import join, exists
import os, sys 
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

#personal modules 
from wofs.data.loadEnsembleData import EnsembleData
from wofs.data.loadEnsembleData import calc_time_max, calc_time_tendency
from wofs.main.forecasts.CalcEnsembleProbabilities import calc_ensemble_probs
from wofs.util import config 
from machine_learning.extraction.StormBasedFeatureEngineering import StormBasedFeatureEngineering
import wofs.util.feature_names as fn 
from wofs.util.basic_functions import convert_to_seconds, personal_datetime
get_time = personal_datetime( )
extract = StormBasedFeatureEngineering( )

""" usage: stdbuf -oL python -u ensemble_feature_extraction.py  2 > & log_extract & """
debug = True

uh_thresh = np.arange(50,260,10)
wnd_thresh = np.arange(30,60,5) 
hail_thresh = np.arange(0.5,1.75,0.25)


def _save_netcdf( data, date, time, fcst_time_idx ):
    '''
    saves a netcdf file
    '''
    ds = xr.Dataset( data )
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars} 
    fname = join(config.ML_INPUT_PATH, str(date), f'PROBABILITY_OBJECTS_{date}-{time}_{fcst_time_idx:02d}.nc')
    #fname = f'PROBABILITY_OBJECTS_{date}-{time}_{fcst_time_idx:02d}.nc'
    print( f"Writing {fname}...")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    ds.to_netcdf( path = fname, encoding=encoding )
    ds.close( )
    del ds, data 
 
##############################
#      MAIN FUNCTION         #
##############################

def worker(date,time,fcst_time_idx):
    """
    worker function for multiprocessing
    """
    duration=6
    time_indexs = np.arange(duration+1)+fcst_time_idx
    print ('\t Starting on {}-{}-{}...'.format(date, time, fcst_time_idx))
    # Read in the probability object file 
    object_file = join(
                    config.OBJECT_SAVE_PATH,
                    date,
                    f'updraft_ensemble_objects_{date}-{time}_t:{fcst_time_idx}.nc'
                  )
    
    if not exists(object_file):
        raise Exception(f'{object_file} does not exist! This is expected since we are using too many times')

    #if exists(fname):
    #    raise Exception(f'{fname} already exists!') 
    
    object_dataset = xr.open_dataset( object_file )
    forecast_objects = object_dataset['Probability Objects'].values
    try:
        ds_subset = object_dataset[fn.probability_object_props]
    except:
        raise Exception(f'KeyError for {date}-{time}-{fcst_time_idx}; No Objects in domain!')
    df = ds_subset.to_dataframe()
    object_labels = df['label']
 
    storm_time_indexs = np.arange(duration+1)+fcst_time_idx
    env_time_indexs = [fcst_time_idx]
    mywofs = EnsembleData( date_dir =date, time_dir = time, base_path ='wofs_data')
    smry = EnsembleData( date_dir =date, time_dir = time, base_path ='summary_files')

    smry_env_data = smry.load_multiple_nc_files(
                                                vars_to_load = fn.new_env_vars_smryfiles,
                                                time_indexs = env_time_indexs,
                                                tag = 'ENV'
                                                )

    smry_storm_data = smry.load_multiple_nc_files(
                                                vars_to_load = fn.new_storm_vars_smryfiles,
                                                time_indexs = storm_time_indexs,
                                                tag = 'ENS'
                                                )
    ###print('fn.new_env_vars_wofsdata', fn.new_env_vars_wofsdata)
    my_env_data = mywofs.load_multiple_nc_files(
                                                vars_to_load = fn.new_env_vars_wofsdata,
                                                time_indexs = env_time_indexs,
                                                tag='DATA'
                                                )

    ###print('fn.new_storm_vars_wofsdata: ', fn.new_storm_vars_wofsdata) 
    my_storm_data = mywofs.load_multiple_nc_files(
                                                vars_to_load = fn.new_storm_vars_wofsdata,
                                                time_indexs = storm_time_indexs,
                                                tag='DATA'
                                                )

    # NEED TO UNCOMMENT FOR NON-2020
    all_env_data = {**smry_env_data, **my_env_data}
    all_strm_data = {**smry_storm_data, **my_storm_data}

    ###all_env_data = smry_env_data 
    ###all_strm_data = smry_storm_data


    time_max_strm_data = {
        var+'_time_max': np.nanmax(all_strm_data[var], axis=0)
        for var in list(all_strm_data.keys()) if var not in fn.new_min_vars
    }

    time_min_strm_data = {
        var+'_time_min': np.nanmin(all_strm_data[var], axis=0)
        for var in fn.new_min_vars
    }

    # Get baseline predictions (80-m wind speed and 2-5 km UH) 
    baseline_uh_probs = {f'uh_probs_>{thresh}': calc_ensemble_probs( time_max_strm_data['uh_2to5_time_max'],
                        thresh ) for thresh in uh_thresh }

    wnd_probs = {f'wnd_probs_>{thresh}': calc_ensemble_probs( time_max_strm_data['ws_80_time_max'],
                        thresh ) for thresh in wnd_thresh }
    
    hail_probs = {f'hail_probs_>{thresh}': calc_ensemble_probs( time_max_strm_data['hailcast_time_max'],
                        thresh ) for thresh in hail_thresh }

    baseline_probs = {**baseline_uh_probs, **wnd_probs, **hail_probs}
    baseline_probs['w_probs'] = object_dataset['2D Probabilities']

    # Combine back the time-max and -min data
    all_strm_data = {**time_max_strm_data, **time_min_strm_data}    
    combined_data = {**all_env_data, **all_strm_data}
    
    ens_mean_data = {
        var+'_ens_mean': np.nanmean(combined_data[var], axis=0)
        for var in list(combined_data.keys())
    }

    ens_std_data = {
        var+'_ens_std': np.nanstd(combined_data[var], axis=0, ddof=1)
        for var in list(combined_data.keys())
    }

    ensemble_data = {**ens_mean_data, **ens_std_data}

    # Extract the features 
    data_ens, ens_feature_names = extract.extract_spatial_features_from_object( ensemble_data, forecast_objects, object_labels, only_mean=True, mem_idx=None )
    data_ens_amp, ens_amp_feature_names = extract.extract_amplitude_features_from_object( all_strm_data, forecast_objects, object_labels)
    
    data_prob, prob_names = extract.extract_spatial_features_from_object( baseline_probs, forecast_objects, object_labels, 
                            only_mean=False, mem_idx=None, stat_funcs=[(np.nanmax,None)], stat_func_names = ['_prob_max'] )

    data = np.concatenate((data_ens, data_ens_amp, data_prob, df.values), axis = 1 )
       
    data = np.array( data )
    data = np.nan_to_num(data) 
    if len(data) > 0:
        #Convert to dictionary with titles for the features
        initialization_time = [convert_to_seconds(time)]*np.shape(data)[0]
        initialization_time  = np.array( initialization_time )
        data = np.concatenate(( data, initialization_time[:, np.newaxis] ), axis = 1)
        feature_names = ens_feature_names + ens_amp_feature_names + prob_names + fn.probability_object_props + fn.additional_vars
        full_data = {var: (['example'], data[:,i]) for i, var in enumerate(feature_names) }
        del data 
        _save_netcdf( data=full_data, date=date, time=time, fcst_time_idx=fcst_time_idx )

# /work/mflora/ML_DATA/INPUT_DATA/20170508/PROBABILITY_OBJECTS_20170508-0130_06.nc
#if debug:
    # 20190530-0130_09.nc
#    worker(date='20180501', time='0130', fcst_time_idx=9)


