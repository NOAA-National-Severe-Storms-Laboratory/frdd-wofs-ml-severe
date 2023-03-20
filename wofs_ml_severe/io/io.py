#==========================================
# Handles the I/O for data and models.
#==========================================
import pandas as pd
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random


class IO:
    
    INFO = ['forecast_time_index', 'obj_centroid_x', 'obj_centroid_y', 'Run Date', 'label']
    
    def __init__(self, basePath = '/work/mflora/ML_DATA/DATA'):
        self.basePath = basePath 
        ###self.outdir = '/work/mflora/ML_DATA/MLDATA'
    
    def _train_test_split(self):
        """
        Randomly split the full dataset into training and testing 
        based on the date. 
        """
        for time in ['first_hour', 'second_hour', 'third_hour', 'fourth_hour']:
            
            path = join(self.basePath, f'wofs_ml_severe__{time}__data.feather')
            print(f'Loading {path}...')
            df = pd.read_feather(path)
    
            # Get the date from April, May, and June 
            df['Run Date'] = df['Run Date'].apply(str)
            df = df[pd.to_datetime(df['Run Date']).dt.strftime('%B').isin(['April', 'May', 'June'])]
            all_dates = list(df['Run Date'].unique())
            random.shuffle(all_dates)
            train_dates, test_dates = train_test_split(all_dates, test_size=0.3)
    
            train_df = df[df['Run Date'].isin(train_dates)]
            test_df  = df[df['Run Date'].isin(test_dates)] 
            
            train_df.reset_index(inplace=True, drop=True)
            test_df.reset_index(inplace=True, drop=True)
            
            print(f'Saving the {time} training and testing datasets...')
            train_df.to_feather(join(self.basePath, f'wofs_ml_severe__{time}__train_data.feather'))
            test_df.to_feather(join(self.basePath, f'wofs_ml_severe__{time}__test_data.feather'))
            
    def get_features(self, df):
        """
        Get the feature columns from the DataFrame. 
        """
        ind = list(df.columns).index('hail_severe_3km')
        non_target_vars = list(df.columns)[:ind]
        features = [f for f in non_target_vars if f not in self.INFO]
        return features
    
    def load_data(self, mode, time, target, return_info=True):
        """Load the training or testing dataframe. """
        path = join(self.basePath, f'wofs_ml_severe__{time}__{mode}_data.feather')
        df = pd.read_feather(path)
        
        features = self.get_features(df)
        X = df[features] 
        y = df[target] 
        
        if return_info:
            info = df[self.INFO]
            return X, y, info

        return X, y
        
    def load_model(self, ):
        """Load ML model"""
        pass
        
        
# Convert New to Old 
def resample_to_old_dataset(df, original_dates):
    dates = df['Run Date'].apply(str)
    return df.loc[dates.isin(original_dates)]

def get_numeric_init_time(X): 
    Xt = X.copy()
    Xt['timestamp'] = pd.to_datetime(Xt['Initialization Time'].values, format='%H%M')
    minutes_since_midnight = lambda x: x.hour 
    Xt['Initialization Time'] = Xt['timestamp'].apply(minutes_since_midnight)
    
    Xt.drop(['timestamp'], axis=1, inplace=True)
    
    return Xt 

def load_ml_data(target_col, 
                 lead_time = 'first_hour', 
                 mode = None, 
                 baseline=False,
                 return_only_df=False, 
                 load_reduced=True, 
                 base_path = '/work/mflora/ML_DATA/DATA',
                ): 
    """ Loads the ML dataset. 
    
    Parameters
    ---------------------
    
    target_col : str or list 
        The target column. If a list, then the different columns are summed together 
        and re-binarized; this is useful for creating all-severe or all-sig-severe targets.
        
    lead_time : 'first_hour', 'second_hour', 'third_hour', or 'fourth_hour'
        The lead time range.
        
    mode : 'training', 'testing' or None (default is None).
        If 'training' or 'testing', the dataset is loaded with the dates from the 
        original dates from the Flora et al. (2021, MWR) paper (for a given hazard 
        and lead time). Otherwise, if None, the full dataset (2017-2021 at the moment) is loaded, 
        but only from dates between March-July. 
        
    Returns
    --------------------
    X : dataframe 
        Input features 
        
    y : 1d array
        Target vector
        
    metadata : dataframe 
        Metadata containing the following data: 
        'Run Date', 'forecast_time_index', 'Initialization Time', 'label', 'obj_centroid_y', 'obj_centroid_x'
    
    """
    # Target Var : [tornado|wind|hail]_[severe|sig_severe]_[3km, 9km, 15km, 30km]_[obj_match | ]
    #base_path = '/work/mflora/ML_DATA/MLDATA'
    #file_path = join(base_path, f'wofs_ml_severe__{lead_time}__{mode}_data.feather')
    
    if baseline:
        if load_reduced:
            file_path = join(base_path, f'wofs_ml_severe__{lead_time}__baseline_reduced_data.feather')
        else:
            file_path = join(base_path, f'wofs_ml_severe__{lead_time}__baseline_data.feather')
    else:
        if load_reduced:
            file_path = join(base_path, f'wofs_ml_severe__{lead_time}__reduced_data.feather')
        else:
            file_path = join(base_path, f'wofs_ml_severe__{lead_time}__data.feather')
    
    df = pd.read_feather(file_path)
 
    if mode is None:
        print('Only keeping warm season cases for the official training!') 
        df = df.loc[pd.to_datetime(
            df['Run Date'].apply(str)).dt.strftime('%B').isin(['March', 'April', 'May', 'June', 'July'])]
    else:
        if isinstance(target_col, list):
            target_col = target_col[0]
            #raise ValueError('At this time, cannot load all severe or all sig severe for the retrospective stuff')
        
        if 'sig' in target_col:
            target = target_col.split('_sig_severe_')[0]
        else:
            target = target_col.split('_severe_')[0]
      
        target = f'severe_{target}' if target != 'tornado' else target
        
        if lead_time in ['third_hour', 'fourth_hour']:
            lead_time = 'second_hour'
        
        dates = pd.read_feather(join('/work/mflora/ML_DATA/DATA', 
                                     'original_dates', f'{mode}_{lead_time}_{target}_dates.feather'))
        original_dates = dates['Dates'].apply(str) 
    
        df = resample_to_old_dataset(df, original_dates)

    # These are the init times to keep. It ignores init times from 1700-1900 (not inclusive). 
    # These are the standard init times used during the spring cases. 
    init_times = ['0000', '0030', '0100', '0130', '0200', '0230', '0300', '1900', '1930', '2000', '2030', '2100',
       '2130', '2200', '2230', '2300', '2330']

    df = df.loc[df['Initialization Time'].isin(init_times)]
    df.reset_index(inplace=True, drop=True) 
    
    # Convert the str init times to hours after midnight. 
    df = get_numeric_init_time(df)
    
    # For the conditional features, replace the NaNs with zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.reset_index(inplace=True, drop=True) 
    
    metadata_features = ['Run Date', 'forecast_time_index', 
                         'Initialization Time', 'label', 'obj_centroid_y', 'obj_centroid_x']
    metadata = df[metadata_features]
    
    features = [f for f in df.columns if 'severe' not in f]
    features = [f for f in features if f not in metadata_features] 
    features.append('Initialization Time')
    
    X = df[features]
         
    if isinstance(target_col, list):
        # Convert to all-severe.
        y = df[target_col].values
        y = np.where(np.sum(y, axis=1)>0, 1, 0)
    else:
        y = df[target_col]

    if return_only_df:
        return df
    else:
        return X,y, metadata     
        
"""
'tornado_severe_0km',
 'tornado_severe_6km',
 'tornado_severe_15km',
 'tornado_severe_original',
 'hail_severe_0km',
 'hail_severe_6km',
 'hail_severe_15km',
 'hail_severe_original',
 'wind_severe_0km',
 'wind_severe_6km',
 'wind_severe_15km',
 'wind_severe_original',
 'tornado_sig_severe_0km',
 'tornado_sig_severe_6km',
 'tornado_sig_severe_15km',
 'tornado_sig_severe_original',
 'hail_sig_severe_0km',
 'hail_sig_severe_6km',
 'hail_sig_severe_15km',
 'hail_sig_severe_original',
 'wind_sig_severe_0km',
 'wind_sig_severe_6km',
 'wind_sig_severe_15km',
 'wind_sig_severe_original',
 'tornado_severe__IOWA_0km',
 'tornado_severe__IOWA_6km',
 'tornado_severe__IOWA_15km',
 'tornado_severe__IOWA_original',
 'hail_severe__IOWA_0km',
 'hail_severe__IOWA_6km',
 'hail_severe__IOWA_15km',
 'hail_severe__IOWA_original',
 'wind_severe__IOWA_0km',
 'wind_severe__IOWA_6km',
 'wind_severe__IOWA_15km',
 'wind_severe__IOWA_original',
 'tornado_sig_severe__IOWA_0km',
 'tornado_sig_severe__IOWA_6km',
 'tornado_sig_severe__IOWA_15km',
 'tornado_sig_severe__IOWA_original',
 'hail_sig_severe__IOWA_0km',
 'hail_sig_severe__IOWA_6km',
 'hail_sig_severe__IOWA_15km',
 'hail_sig_severe__IOWA_original',
 'wind_sig_severe__IOWA_0km',
 'wind_sig_severe__IOWA_6km',
 'wind_sig_severe__IOWA_15km',
 'wind_sig_severe__IOWA_original'
"""
        