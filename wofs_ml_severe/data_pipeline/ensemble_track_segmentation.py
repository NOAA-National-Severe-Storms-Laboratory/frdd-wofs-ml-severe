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
_base_wofs_path = '/home/monte.flora/python_packages/WoF_post'
_base_mp_path = '/home/monte.flora/python_packages/MontePython'
import sys
sys.path.insert(0,_base_mp_path)
sys.path.insert(0,_base_wofs_path)

from monte_python.object_identification import label_per_member
import monte_python 

from wofs.post.utils import save_dataset, load_yaml
from wofs.common.zarr import open_dataset, normalize_filename
from wofs.common import remove_reserved_keys

from ..conf.segmentation_config import config


def identify_ensemble_tracks(deterministic_tracks, 
                             params, ensemble_size, 
                             split_regions=True):
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
        
    Returns
    ------------
    storm_labels : array of shape (NY, NX)
    ensemble_probabilities: array of shape (NY, NX)
    """
    # Compute the grid-scale ensemble probability of exceedance. 
    deterministic_tracks_binarized = np.where(deterministic_tracks > 0, 1, 0)
    ensemble_probabilities = np.average(deterministic_tracks_binarized, axis=0)

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
        
    # Identify the tracks. 
    storm_labels, object_props = monte_python.label(input_data = new_input_data, 
                       method ='iterative_watershed', 
                       return_object_properties=True, 
                       params = params,  
                       )
    
    """
    # Add QC 
    qcer = monte_python.QualityControler()
    if split_regions: 
        qc_params = [('max_area_before_split', 1000), ('trim', (6/18, 4/18) )]
    else:
        qc_params = [('trim', (6/18, 4/18) )]
    
    try:
        storm_labels, _ = qcer.quality_control(ensemble_probabilities, storm_labels, object_props, qc_params)
    except AssertionError:
        print('Splitting failed!')
        qc_params = [('trim', (6/18, 4/18) )]
        storm_labels, _ = qcer.quality_control(ensemble_probabilities, storm_labels, object_props, qc_params)
    """
    
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
            if np.ndim(vals) == 3:
                ds[v] = (['NE', 'NY', 'NX'], vals)
            else:
                ds[v] = (['NY', 'NX'], vals)
        
        # Save the original dataset
        return save_dataset(fname=file_path, dataset=ds)
    
    # If the file doesn't exist, then just save the new_ds 
    return save_dataset(fname=file_path, dataset=new_ds)
    
def generate_ensemble_track_file(ncfile, outdir=None, overwrite=True, 
                                 debug=False,  **kwargs):
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
    deterministic_tracks = label_per_member(
                            data_to_label=data_to_label, 
                            method=config['deterministic'][f'params'][0], 
                            params=config['deterministic'][f'params'][1], 
                            qc_params=config['deterministic'][f'qc_params']
                            )

    # Identify the ensemble storm tracks. 
    results = identify_ensemble_tracks(deterministic_tracks,
                                           config['ensemble'][f'params'],
                                           ensemble_size=config['ensemble']['ensemble_size'] 
                                      )
    
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