#======================================================
# This is a script to split the ML data set into training
# and testing for both the full ML and baseline
# datasets.
# 
# Updates: Lucas Jones (Git username : LucasJ-NSSL)
# Email: lucas.jones@noaa.gov
# Date: Sept. 28, 2026
#======================================================

import numpy as np 
from tqdm import tqdm
from os.path import join
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import json

def train_test_splitter(months =['April', 'May', 'June', 'July'],
                        test_size=0.3, baseline = True):
    """
    Randomly split the full ML and BL datasets into training and testing 
    based on the date. The testing dataset size is based on 
    test_size, which determines the percentage of cases set aside for 
    testing. 
    """

    BASE_PATH = '/work2/lucas.jones/ml_data/'
    
    for time in tqdm(['first_hour', 'second_hour', 'third_hour', 'fourth_hour']):
        path = join(BASE_PATH, f'wofs_ml_severe__{time}__data.feather')
        df = pd.read_feather(path)
    
        print(f'Full Dataset Shape: {df.shape=}')
        
        # Get the date from April, May, and June 
        df['Run Date'] = df['Run Date'].apply(str)
        
        df = df[pd.to_datetime(df['Run Date']).dt.strftime('%B').isin(months)]

        all_dates = list(df['Run Date'].unique())
        random.shuffle(all_dates)
        train_dates, test_dates = train_test_split(all_dates, test_size=test_size)

        train_json = json.dumps(train_dates, indent = 0)
        test_json = json.dumps(test_dates, indent = 0)

        with open(join(BASE_PATH, f'wofs_ml_severe__{time}__training_dates.json'), "w") as file:
            file.write(train_json)

        with open(join(BASE_PATH, f'wofs_ml_severe__{time}__testing_dates.json'), "w") as file:
            file.write(test_json)
    
        '''
        train_df = df[df['Run Date'].isin(train_dates)] 
        test_df  = df[df['Run Date'].isin(test_dates)] 
    
        print(f'Training Dataset Size: {train_df.shape=}')
        print(f'Testing  Dataset Size: {test_df.shape=}')
    
        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)
        
        train_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{time}__train_data.feather'))
        test_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{time}__test_data.feather'))

        if baseline:
            print("Baseline train/test split activated")

            baseline_path = join(BASE_PATH, f'wofs_ml_severe__{time}__baseline_data.feather')
            baseline_df = pd.read_feather(baseline_path)

            # Get the date from April, May, and June 
            baseline_df['Run Date'] = baseline_df['Run Date'].apply(str)
            
            baseline_df = baseline_df[
                pd.to_datetime(baseline_df['Run Date']).dt.strftime('%B').isin(months)]

            train_base_df = baseline_df[baseline_df['Run Date'].isin(train_dates)] 
            test_base_df  = baseline_df[baseline_df['Run Date'].isin(test_dates)] 
  
            train_base_df.reset_index(inplace=True, drop=True)
            test_base_df.reset_index(inplace=True, drop=True)
        
            train_base_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{time}__train_baseline_data.feather'))
            test_base_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{time}__test_baseline_data.feather'))

        else:
            print("Note: baseline train/test split not activated")
        '''

# switch to determine if baseline splitting should be performed also
baseline = False

train_test_splitter(baseline = baseline)

print("Date based train/test splitting completed!")