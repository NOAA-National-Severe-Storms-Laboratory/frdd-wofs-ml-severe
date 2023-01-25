#======================================================
# Python script for the ensemble storm track segementation
# 
# Author: Montgomery Flora (Git username : monte-flora)
# Email : monte.flora@noaa.gov 
#======================================================

# Python Modules 
import os, pathlib

# Third Party Modules 
import xarray as xr 
import numpy as np 
from scipy.ndimage import maximum_filter, gaussian_filter

# WoFS modules 
_base_module_path = '/home/monte.flora/python_packages/master/WoF_post'
import sys
sys.path.append(_base_module_path)

import monte_python
from wofs.post.wofs_cbook import identify_deterministic_tracks
from wofs.post.utils import save_dataset, load_yaml
from wofs.common.zarr import open_dataset, normalize_filename
from wofs.common import remove_reserved_keys

from ..conf.segmentation_config import config

#from wofs.post.multiprocessing_script import run_parallel_realtime, to_iterator

# python wofs_probability_tracks.py 
# -i /scratch/brian.matilla/WoFS_2020/summary_files/WOFS_RLT/20200427/3KM/2100 -d 5 -n 24 --nt 12 --duration 30 


def identify_ensemble_tracks(deterministic_tracks, params, ensemble_size, remove_low=True, previous_method=False):
    """
    Procedure for the ensemble storm track segmentation.
    
    Parameters
    -----------
    deterministic_tracks : 3D array (NE, NY, NX)
        An ensemble of labeled updraft tracks. 
    
    params : list of dicts
        Parameters for the iterative watershed method
    
    ensemble_size : int 
        Ensemble size 
        
    remove_low : bool
        If True, remove tracks with probabilities <= 0.12 (3 ensemble members)
    
    
    Returns
    ------------
    storm_labels : array of shape (NY, NX)
    ensemble_probabilities: array of shape (NY, NX)
    """
    # Compute the grid-scale ensemble probability of exceedance. 
    deterministic_tracks_binarized = np.where(deterministic_tracks > 0, 1, 0)
    ensemble_probabilities = np.average(deterministic_tracks_binarized, axis=0)

    if previous_method:
        # These are settings from the 2021 paper and those used in real-time in 2020 and 2021.
        new_input_data = monte_python.object_identification.quantize_probabilities(
            np.copy(ensemble_probabilities), ensemble_size)

        param_set = [ {'min_thresh': 0,
                   'max_thresh': 18,
                   'data_increment': 1,
                   'delta': 0,
                   'area_threshold': 400,
                   'dist_btw_objects': 15 },

                  {'min_thresh': 5,
                   'max_thresh': 18,
                   'data_increment': 1,
                   'delta': 0,
                   'area_threshold': 300,
                   'dist_btw_objects': 25 }
            ]

        params = {'params': param_set }
        
    else:
        # Create a copy of the data .
        new_input_data = np.copy(ensemble_probabilities)
    
        # Apply a maximum-val filter to smooth the probabilities.
        new_input_data = maximum_filter(new_input_data, size=4)
    
        # Remove tracks with low probabilities.
        if remove_low:
            new_input_data[ensemble_probabilities<=0.12] = 0
        
        # Apply a minimal gaussian filter for additional smoothing.
        new_input_data = gaussian_filter(new_input_data, 1.5)*100

    
    # Identify the tracks. 
    storm_labels = monte_python.label(input_data = new_input_data, 
                       method ='iterative_watershed', 
                       return_object_properties=False, 
                       params = params,  
                       )
    
    if not previous_method:
        # Reduce the object size due to the maximum filter and gaussian filter 
        idx = np.where(ensemble_probabilities==0)
        storm_labels[idx] = 0
    
    return storm_labels, ensemble_probabilities
 
def version_control_procedure(file_path, data_vars, new_ds):
    """
    If a file already exists, we can re-write existing variables 
    with a version-based naming instead of completely overwriting 
    the file. 
    
    Parameters
    ------------------
    file_path : path-like, str
        Path to the file that will be re-written.
    
    data_vars : list of strs
        Variables to get a version-based naming and won't be re-written
        
    new_ds : xarray.Dataset 
    
    
    """
    # If the file exists, then the variables will need to overwritten. 
    if os.path.exists(file_path):
        ds = xr.load_dataset(file_path, decode_times=False)
        all_data_vars = ds.data_vars
        for v in data_vars: 
            var_subset = [vr for vr in all_data_vars if v in vr]
            ###print(f'{var_subset=}')
            # Start with the newest old one.
            var_subset.sort(reverse=True)
            var_subset = var_subset
            # Do we already have existing all versions of this variable?
            # Reset the naming. 
            var_subset.insert(0, f'{v}__v{len(var_subset)}')
            for i in range(len(var_subset[:-1])):
                ##print(var_subset[i], var_subset[i+1])
                ds[var_subset[i]] = (['NY', 'NX'], ds[var_subset[i+1]].values)  
        
            # Finally, set the original variable name to the data from the
            # new dataset. 
            vals = new_ds[v].values
            print(v, vals.shape, np.ndim(vals))
            if np.ndim(vals) == 3:
                ds[v] = (['NE', 'NY', 'NX'], vals)
            else:
                ds[v] = (['NY', 'NX'], vals)
        
        # Save the original dataset
        return save_dataset(fname=file_path, dataset=ds)
    
    # If the file doesn't exist, then just save the new_ds 
    return save_dataset(fname=file_path, dataset=new_ds)
    
def generate_ensemble_track_file(ncfile, outdir=None, overwrite=True, debug=False, previous_method=True, **kwargs):
    """
    Generates the ENSEMBLETRACK summary file. 
    
    Parameters
    ---------------
    ncfile : path-like, str
        A path to a 30M summary file. This summary files contains 
        the deterministic 30-min updraft tracks used to identify 
        the ensemble storm tracks.
        
    overwrite : bool (default=True)
        If True, if the file already exists, it will be overwritten
        If False, if the file already exists, then the variable names
        are altered and then the new data is appended to the file. 
    
    keyword arg can include 'output_file' 
        
    """
    VAR = 'w_up'
    
    # Open the netCDF file. 
    try:
        ds = open_dataset(ncfile, decode_times=False)
    except OSError:
        print(f'Unable to open {ncfile}!') 
        return ncfile

    data_to_label = ds[VAR].values
    
    # Identify the deterministic 30-min updraft tracks. 
    deterministic_tracks = identify_deterministic_tracks(
                            data_to_label=data_to_label, 
                            method=config['deterministic'][f'params'][0], 
                            params=config['deterministic'][f'params'][1], 
                            qc_params=config['deterministic'][f'qc_params']
                            )

    # Identify the ensemble storm tracks. 
    results = identify_ensemble_tracks(deterministic_tracks,
                                           config['ensemble'][f'params'],
                                           ensemble_size=config['ensemble']['ensemble_size'], 
                                      previous_method=previous_method)
        
    field_names = [f'{VAR}__ensemble_tracks', f'{VAR}__ensemble_probabilities']
    data = {field: (['NY','NX'], result) for field, result in zip(field_names, results) }
    data['updraft_tracks'] = (['NE', 'NY', 'NX'], deterministic_tracks)
    
    
    # Convert the data to xarray.Dataset.
    dataset = xr.Dataset(data)
    
    dataset['xlat'] = ds['xlat']
    dataset['xlat'].attrs = remove_reserved_keys(ds['xlat'].attrs)

    dataset['xlon'] = ds['xlon'] 
    dataset['xlon'].attrs = remove_reserved_keys(ds['xlon'].attrs)

    # copy global attributes.
    dataset.attrs = remove_reserved_keys(ds.attrs)
    
    # Close the dataset. 
    ds.close()
    
    # Save the dataset.
    save_filename = kwargs.pop('output_file', 
                               normalize_filename(
                                   ncfile).replace('30M','ENSEMBLETRACKS').replace('scratch','work').replace('_RLT','_ML'))
    if outdir is not None:
        if not os.path.exists:
            os.makedirs(outdir)
        save_filename = os.path.join(outdir,save_filename) 
    
    if debug:
        save_filename = os.path.basename(save_filename).replace('.json', '.nc')
        save_filename = os.path.join(outdir, save_filename)
        print(f'Saving {save_filename}...')
        save_dataset(fname=save_filename, dataset=dataset)
        return save_filename
        
        
    if overwrite:
        # In this case, we want to overwrite the file.
        try:
            save_dataset(fname=save_filename, dataset=dataset)
        except PermissionError:
            print(f"PermissionError with {save_filename}! May need to ask Brian for permissions")
    else:
        version_control_procedure(save_filename, dataset.data_vars, dataset)
    
    return save_filename


'''
Deprecated. 

def generate_ensemble_track_dir(indir, dt, n_processors, nt, duration, runtype):
    """
    This script will generate the ensemble storm tracks for updraft, and mid- and low-level UH.

    Parser Arguments: 
    ------------------------
        -i, --indir : str
            file path to the series of ENS summary files 
            to be processed
        -d, --dt : int
            the forecast interval between ENS summary files
            (this variable should be declared in a config.yaml
            in the Wof_post/conf)
        -n, --n_processors : int or float
            Number of processors to use or
            percentage of the processors to use
        --nt : int
            Number of time steps to process
        --duration : int
            Duration of the track (in minutes)
        --runtype : 'rto' or 'rlt'
            Whether the script is being ran in research mode ('rto')
            or in realtime ('rlt'). If 'rlt', the scripts are delayed
            to allow for WRFOUT files to be generated before
            processing the summary files.
    """

    delta_time_step = int(duration / dt)
    total_idx = (nt + delta_time_step) + 1
    rtype = runtype

    summary_file_dir = indir
    summary_ens_files = generate_summary_file_name(
        indir=indir,
        outdir=indir,
        time_idxs=range(total_idx),
        mode="ENS",
        output_timestep=dt,
        first_hour=14,
    )

    if duration == 30:
        if nt == 72 or nt == 36:
            nt = (nt - delta_time_step) + 1
        else:
            nt = nt + 1
        iterator = range(nt)
    elif duration == 60:
        iterator = range(0, nt, delta_time_step)

    iterator_size = len(list(iterator))

    files = []
    for i in iterator:
        nc_file_paths = [
            join(summary_file_dir, f)
            for f in summary_ens_files[i : i + delta_time_step + 1]
        ]
        files.append(nc_file_paths)


    files_to_load =[]
    for nc_file_paths in files:
        fname = generate_track_filename(
            nc_file_paths[0], nc_file_paths[-1], duration=duration, nt=nt
        )
        files_to_load.append(join(summary_file_dir, fname))

    generate_ensemble_track_files(files_to_load, n_processors, rtype)


def generate_ensemble_track_files(files, nprocs, rtype):
    """
    files array should be a list of summary files to load in parallel
    """
    run_parallel_realtime(
        func=generate_ensemble_track_file,
        nprocs_to_use=nprocs,
        iterator=to_iterator(files),
        rtype=rtype
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir", type=str)
    parser.add_argument("-d", "--dt", type=int)
    parser.add_argument("-n", "--n_processors", type=float)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--runtype", type=str)

    args = parser.parse_args()

    generate_ensemble_track_dir(args.indir, args.dt, args.n_processors, args.nt, args.duration, args.runtype)
'''
