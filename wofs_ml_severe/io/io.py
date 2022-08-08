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
    
    
    def _train_test_split(self):
        """
        Randomly split the full dataset into training and testing 
        based on the date. 
        """
        for time in ['first_hour', 'second_hour']:

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
        
        
def load_ml_data(lead_time = 'first_hour', 
                 mode = 'train',
                 cols_to_drop = ['label', 'obj_centroid_x', 
                                 'obj_centroid_y', 'Run Date', 
                                 'forecast_time_index'], 
                target_col = 'hail_severe_3km_obj_match',
                sanity_check=False 
                ): 
    """ Loads the ML dataset. """
    # Target Var : [tornado|wind|hail]_[severe|sig_severe]_[3km, 9km, 15km, 30km]_[obj_match | ]
    #base_path = '/work/mflora/ML_DATA/MLDATA'
    #file_path = join(base_path, f'wofs_ml_severe__{lead_time}__{mode}_data.feather')
    
    if sanity_check:
        base_path = '/work/mflora/ML_DATA/DATA'
        file_path = join(base_path, f'wofs_ml_severe__{lead_time}__data.feather')
    else:
        base_path = '/work/mflora/ML_DATA/MLDATA'
        file_path = join(base_path, f'wofs_ml_severe__{lead_time}__{mode}_data.feather')
        
    df = pd.read_feather(file_path)

    metadata = df[['Run Date', 'forecast_time_index', 'Initialization Time', 'label']]
    index = list(df.columns).index('hail_severe_3km_obj_match')
    possible_features = list(df.columns)[:index]

    drop_vars = ['QVAPOR', 'freezing_level', 'stp', 'okubo_weiss', 'Initialization Time', 
                 'qv_2', 'srh_0to500'
                ]
    
    features = [f for f in possible_features if f not in cols_to_drop]
    
    if sanity_check:
        features = [f for f in features if not any([d in f for d in drop_vars])]
    
    X = df[features]
    y = df[target_col]

    return X,y, metadata     
        
        
        