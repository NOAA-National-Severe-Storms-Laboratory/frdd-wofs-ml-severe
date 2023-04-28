import re 
from os.path import exists, basename
import numpy as np


def fix_data(X): 
    #X = X.astype({'Initialization Time' : str})
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.reset_index(inplace=True, drop=True)
    
    return X  

# Temporarily Deprecated until new models are generated
"""
def get_time_str(time_index): 
    hr_size = 12 
    blend_period = 5
    hr_rng = np.arange(13)
    ranges = {'first_hour' : hr_rng, 
              'second_hour' : hr_rng+hr_size, 
              'third_hour' : hr_rng+(2*hr_size), 
              'fourth_hour': hr_rng+(3*hr_size)
             }

    if time_index in ranges['first_hour']:
        time = 'first_hour'
        
    elif time_index in ranges['second_hour'][:blend_period]:
        time = ['first_hour', 'second_hour']
    elif time_index in ranges['second_hour'][blend_period:]:
        time = 'second_hour'
        
    elif time_index in ranges['third_hour'][:blend_period]:
        time = ['second_hour', 'third_hour']
    elif time_index in ranges['third_hour'][blend_period:]:
        time = 'third_hour'
    elif time_index in ranges['fourth_hour'][:blend_period]:
         time = ['third_hour', 'fourth_hour']
    else:
        time = 'fourth_hour'
        
    return time
"""
def get_time_str(ts):
    # 12 is for the 12 timesteps in the first hour (dt=5min)
    first_hour = 12
    # blending the first and second hour output for 5 timesteps 
    blend_period = 4
    if ts <= first_hour:
        time = 'first_hour'
    elif first_hour < ts <= first_hour+blend_period:
        time = ['first_hour', 'second_hour']
    else:
        time = 'second_hour'
        
    return time 


def is_list(a):
    return isinstance(a, list)


def get_target_str(target):
    if 'all' in target:
        return target 
    else:
        comps = target.split('_')
        hazard = comps[0]
        if 'sig' in target:
            return f'{hazard}_sig'
        
    return hazard


def isPath(s):
    """
    @param s string containing a path or url
    @return True if it's a path, False if it's an url'
    """
    if exists(s): 
        return True
    elif s.startswith("/"): 
        return True
    elif len(s.split("/")) > 1: 
        return True
    else:
        return False

def save_dataset(fname, dataset, complevel=5):
    """ saves xarray dataset to netcdf """
    comp = dict(zlib=True, complevel=complevel)
    encoding = {var: comp for var in dataset.data_vars}
    #os.makedirs(os.path.dirname(fname), exist_ok=True)
    dataset.to_netcdf( path = fname, encoding=encoding )
    dataset.close( )
    del dataset


def decompose_file_path(file_path, 
                        file_pattern = 'wofs', 
                        comp_names = None,
                        decompose_path = False, 
                       ):
    """
    Decompose a file into its components. Default behavior is to decompose 
    WoFS summary files into components, but could be used for other file paths.
    
    Parameters
    ----------------
    file_path : 'wofs', 'wrfin', str, path-like 
        Path to a file or the filename. If a path, then the code internally converts to 
        the file name. 
    
    file_pattern : re-based str
        A re-structured string 
        
    comp_names : list of strings
        Names of the components
        
    decompose_path : True/False (default=False)
        If True, then decompose_file_path assumes that file_pattern is a path-like string
        otherwise, the decompose_file_path will treat it as the file name. 
    
    Returns
    -------------
        components : dict 
            A dictionary of file component names and the components themselves. 
    
    Raises
    ------------
    ValueError
        Components names must be provide, if the user provides a file path (not using 
        one of the default options) 
    
    ValueError
        The given file must match the pattern given.
        
    AssertionError
        The Number of components has to equal the number of file path components. 
    
    """
    if file_pattern == 'wofs':
        file_pattern = 'wofs_(\S{3,14})_(\d{2,3})_(\d{8})_(\d{4})_(\d{4}).(nc|json|feather)'
        comp_names = ['TYPE', 'TIME_INDEX', 'VALID_DATE', 'INIT_TIME', 'VALID_TIME', 'FILE_TYPE']
        
        
    if not decompose_path:
        if isPath(file_path):
            file_path = basename(file_path)
            if comp_names is None:
                raise ValueError('Must provide names for the file path components!') 
    
    
    dtre = re.compile(file_pattern)
    
    try:
        obj = dtre.match(file_path)
    except:
        raise ValueError('File given does not match the pattern!') 
    
    if obj is None:
        raise ValueError('File given does not match the pattern!') 
    
    comps = obj.groups()
    
    assert len(comps) == len(comp_names), f"""
                                          Number of component names does not equal the number of file components!
                                          components: {comps} 
                                          component names : {comp_names}
                                          """
    
    components = {n : c for n,c in zip(comp_names, comps)}
    
    return components 