import re 
from os.path import exists, basename, dirname
import numpy as np
import pandas as pd 
from datetime import datetime, timedelta


def get_init_time(filename):
    return basename(dirname(filename))
                   
def get_valid_time(filename, offset=6, dt=5):
    comps = decompose_file_path(filename)

    init_time = comps['VALID_DATE']+get_init_time(filename)

    valid_duration = int(comps['TIME_INDEX'])*dt - (offset*dt)
    start_time=(pd.to_datetime(init_time)+timedelta(minutes=valid_duration)).strftime('%Y%m%d%H%M')
    
    return start_time




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


def get_target_str(target, return_altered=True):
    if return_altered:
        if 'all' in target:
            return target 
        else:
            comps = target.split('_')
            hazard = comps[0]
            if 'sig' in target:
                return f'{hazard}_sig'
        
        return hazard
    else:
        return target 


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

#from __future__ import absolute_import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

#determine unconditional mean, sum R in each bin. But then devide by master counts
def boxbin(x,y,xedge,yedge,c=None,figsize=(5,5),cmap='viridis',
           mincnt=10,vmin=None,vmax=None,edgecolor=None,powernorm=False,
           ax=None,normed=False,method='mean',quantile=None,alpha=1.0,cbar=True,
           unconditional=False,master_count=np.array([])):
    
    """ This function will grid data for you and provide the counts if no variable c is given, or the median if 
    a variable c is given. In the future I will add functionallity to do the median, and possibly quantiles. 
    
    x: 1-D array 
    y: 1-D array 
    xedge: 1-D array for xbins 
    yedge: 1-D array for ybins
    
    c: 1-D array, same len as x and y 
    
    returns
    
    axis handle 
    cbar handle 
    C matrix (counts or median values in bin)
    
    """
    
    midpoints = np.empty(xedge.shape[0]-1)
    for i in np.arange(1,xedge.shape[0]):
        midpoints[i-1] = xedge[i-1] + (np.abs(xedge[i] - xedge[i-1]))/2.
    
    #note on digitize. bin 0 is outside to the left of the bins, bin -1 is outside to the right
    ind1 = np.digitize(x,bins = xedge) #inds of x in each bin
    ind2 = np.digitize(y,bins = yedge) #inds of y in each bin
    
    
    #drop points outside range 
    outsideleft = np.where(ind1 != 0)
    ind1 = ind1[outsideleft]
    ind2 = ind2[outsideleft]
    if c is None:
        pass
    else:
        c = c[outsideleft]
        
    outsideright = np.where(ind1 != len(xedge))
    ind1 = ind1[outsideright]
    ind2 = ind2[outsideright]
    if c is None:
        pass
    else:
        c = c[outsideright]
        
    outsideleft = np.where(ind2 != 0)
    ind1 = ind1[outsideleft]
    ind2 = ind2[outsideleft]
    if c is None:
        pass
    else:
        c = c[outsideleft]
    outsideright = np.where(ind2 != len(yedge))
    ind1 = ind1[outsideright]
    ind2 = ind2[outsideright]
    if c is None:
        pass
    else:
        c = c[outsideright]
    

    if c is None:
        c = np.zeros(len(ind1))
        df = pd.DataFrame({'x':ind1-1,'y':ind2-1,'c':c})
        df2 = df.groupby(["x","y"]).count()
        df = df2.where(df2.values >= mincnt).dropna()
        C = np.ones([xedge.shape[0]-1,yedge.shape[0]-1])*-9999
        for i,ii in enumerate(df.index.values):
            C[ii[0],ii[1]] = df.c.values[i]
        C = np.ma.masked_where(C == -9999,C)
        
        if normed:
            n_samples = np.ma.sum(C)
            C = C/n_samples
            C = C*100
            print('n_samples= {}'.format(n_samples))
        
        if ax is None:
            fig = plt.figure(figsize=(5,5))
            ax = plt.gca()
        else:
            pass
            
        if powernorm:
            pm = ax.pcolormesh(xedge,yedge,C.transpose(),cmap=cmap,edgecolor=edgecolor,norm=colors.PowerNorm(gamma=0.5),vmin=vmin,vmax=vmax,alpha=alpha)
            
            if cbar:
                cbar = plt.colorbar(pm,ax=ax)
            else:
                cbar = pm 
        else:
            pm = ax.pcolormesh(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,edgecolor=edgecolor,alpha=alpha)
            if cbar:
                cbar = plt.colorbar(pm,ax=ax)
            else:
                cbar = pm 
            
        return ax,cbar,C
    
    elif unconditional:
    
        df = pd.DataFrame({'x':ind1-1,'y':ind2-1,'c':c})
        if method=='mean':
            df2 = df.groupby(["x","y"])['c'].sum()
            
        df3 = df.groupby(["x","y"]).count()
        df2 = df2.to_frame()
        df2.insert(1,'Count',df3.values)
        df = df2.where(df2.Count >= mincnt).dropna()
        C = np.ones([xedge.shape[0]-1,yedge.shape[0]-1])
        for i,ii in enumerate(df.index.values):
            C[ii[0],ii[1]] = df.c.values[i]
                
        C = C/master_count.values

        if ax is None:
            fig = plt.figure(figsize=(5,5))
            ax = plt.gca()
        else:
            pass
        
        if powernorm:
            pm = ax.pcolor(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,norm=colors.PowerNorm(gamma=0.5),alpha=alpha)
            if cbar:
                cbar = plt.colorbar(pm,ax=ax)
        else:
            
            pm = ax.pcolor(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
            if cbar: 
                cbar = plt.colorbar(pm,ax=ax)
        
        
    else:
        df = pd.DataFrame({'x':ind1-1,'y':ind2-1,'c':c})
        if method=='mean':
            df2 = df.groupby(["x","y"])['c'].mean()
        elif method=='std':
            df2 = df.groupby(["x","y"])['c'].std()
        elif method=='median':
            df2 = df.groupby(["x","y"])['c'].median()
        elif method=='qunatile':
            if quantile is None:
                print('No quantile given, defaulting to median')
                quantile = 0.5
            else:
                pass
            df2 = df.groupby(["x","y"])['c'].apply(percentile(quantile*100))
            
            
        df3 = df.groupby(["x","y"]).count()
        df2 = df2.to_frame()
        df2.insert(1,'Count',df3.values)
        df = df2.where(df2.Count >= mincnt).dropna()
        C = np.ones([xedge.shape[0]-1,yedge.shape[0]-1])*-9999
        for i,ii in enumerate(df.index.values):
            C[ii[0],ii[1]] = df.c.values[i]

        C = np.ma.masked_where(C == -9999,C)

        if ax is None:
            fig = plt.figure(figsize=(5,5))
            ax = plt.gca()
        else:
            pass
        
        if powernorm:
            pm = ax.pcolor(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,norm=colors.PowerNorm(gamma=0.5),alpha=alpha)
            if cbar:
                cbar = plt.colorbar(pm,ax=ax)
            else:
                cbar = pm
        else:
            
            pm = ax.pcolor(xedge,yedge,C.transpose(),cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
            if cbar: 
                cbar = plt.colorbar(pm,ax=ax)
            else:
                cbar = pm 
            
    return ax,cbar,C
