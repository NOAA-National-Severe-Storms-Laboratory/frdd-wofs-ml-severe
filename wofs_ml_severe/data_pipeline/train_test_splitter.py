import numpy as np 
from tqdm import tqdm

# Split the ML dataset into training and testing. 

def train_test_splitter(months =['March', 'April', 'May', 'June', 'July'],
                        test_size=0.3):
    """
    Randomly split the full ML and BL datasets into training and testing 
    based on the date. The testing dataset size is based on 
    test_size, which determines the percentage of cases set aside for 
    testing. 
    """
    BASE_PATH = '/work/mflora/ML_DATA/DATA'
    OUT_PATH = '/work/mflora/ML_DATA/MLDATA'
    
    for time in tqdm(['first_hour', 'second_hour', 'third_hour', 'fourth_hour']):
        path = join(BASE_PATH, f'wofs_ml_severe__{time}__data.feather')
        df = pd.read_feather(path)
    
        print(f'Full Dataset Shape: {df.shape=}')
    
        baseline_path = join(BASE_PATH, f'wofs_ml_severe__{time}__baseline_data.feather')
        baseline_df = pd.read_feather(baseline_path)
        
        # Get the date from April, May, and June 
        df['Run Date'] = df['Run Date'].apply(str)
        baseline_df['Run Date'] = baseline_df['Run Date'].apply(str)
        
        df = df[pd.to_datetime(df['Run Date']).dt.strftime('%B').isin(months)]
        baseline_df = baseline_df[
            pd.to_datetime(baseline_df['Run Date']).dt.strftime('%B').isin(months)]
        
        all_dates = list(df['Run Date'].unique())
        random.shuffle(all_dates)
        train_dates, test_dates = train_test_split(all_dates, test_size=test_size)
    
        train_df = df[df['Run Date'].isin(train_dates)] 
        test_df  = df[df['Run Date'].isin(test_dates)] 
    
        train_base_df = baseline_df[baseline_df['Run Date'].isin(train_dates)] 
        test_base_df  = baseline_df[baseline_df['Run Date'].isin(test_dates)] 
    
        print(f'Training Dataset Size: {train_df.shape=}')
        print(f'Testing  Dataset Size: {test_df.shape=}')
    
        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)
        
        train_base_df.reset_index(inplace=True, drop=True)
        test_base_df.reset_index(inplace=True, drop=True)
        
        train_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{time}__train_data.feather'))
        test_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{time}__test_data.feather'))
        
        train_base_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{time}__train_baseline_data.feather'))
        test_base_df.to_feather(join(OUT_PATH, f'wofs_ml_severe__{time}__test_baseline_data.feather'))
