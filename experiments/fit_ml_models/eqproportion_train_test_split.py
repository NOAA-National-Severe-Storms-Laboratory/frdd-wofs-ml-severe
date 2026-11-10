import numpy as np 
from tqdm import tqdm
from os.path import join
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def train_test_splitter(months=['April', 'May', 'June', 'July'],
                        test_size=0.1, file_version=None, baseline=True):
    
    BASE_PATH = '/work2/lucas.jones/ml_data/'
    OUT_PATH = BASE_PATH


    SEVERE_HAIL_THRESH = 1.0
    SEVERE_WIND_THRESH = 50.0
    SEVERE_TORN_THRESH = 1
    
    print("Generating uniform train/test split across all lead times...")
    
    # =====================================================================
    # 1. AGGREGATE TARGETS TO THE DATE LEVEL
    # =====================================================================
    temp_path = join(BASE_PATH, 'wofs_ml_severe__first_hour__data.feather')
    temp_df = pd.read_feather(temp_path)
    
    temp_df['Run Date'] = temp_df['Run Date'].apply(str)
    temp_df = temp_df[pd.to_datetime(temp_df['Run Date']).dt.strftime('%B').isin(months)]
    
    # 1 if hazard occurred at least once on that row, 0 otherwise
    temp_df['is_hail'] = np.where(temp_df['hail_any_0km'] >= SEVERE_HAIL_THRESH, 1, 0)
    temp_df['is_wind'] = np.where(temp_df['wind_any_0km'] >= SEVERE_WIND_THRESH, 1, 0)
    temp_df['is_torn'] = np.where(temp_df['tornado_any_0km'] >= SEVERE_TORN_THRESH, 1, 0)
    
    # Max per day: 1 if it occurred at all on that date, 0 if not
    date_groups = temp_df.groupby('Run Date')[['is_hail', 'is_wind', 'is_torn']].max()
    unique_dates = date_groups.index.tolist()
    
    # =====================================================================
    # 2. RANDOM SEARCH FOR OPTIMAL DATE RATIOS
    # =====================================================================
    # Calculate the global ratio of dates with an event to total dates
    global_prop_hail = date_groups['is_hail'].mean()
    global_prop_wind = date_groups['is_wind'].mean()
    global_prop_torn = date_groups['is_torn'].mean()
    
    best_error = float('inf')
    train_dates = None
    test_dates = None
    
    # Test 10,000 random splits to find the one where the train/test proportions 
    # best match the global proportions
    for seed in range(10000):
        trn_d, tst_d = train_test_split(unique_dates, test_size=test_size, random_state=seed)
        
        # Calculate the ratio of dates with an event in the training partition
        trn_prop_hail = date_groups.loc[trn_d, 'is_hail'].mean()
        trn_prop_wind = date_groups.loc[trn_d, 'is_wind'].mean()
        trn_prop_torn = date_groups.loc[trn_d, 'is_torn'].mean()
        
        # Calculate the ratio of dates with an event in the testing partition
        tst_prop_hail = date_groups.loc[tst_d, 'is_hail'].mean()
        tst_prop_wind = date_groups.loc[tst_d, 'is_wind'].mean()
        tst_prop_torn = date_groups.loc[tst_d, 'is_torn'].mean()
        
        # Compute absolute difference from the global ratio for both partitions
        err_hail = abs(trn_prop_hail - global_prop_hail) + abs(tst_prop_hail - global_prop_hail)
        err_wind = abs(trn_prop_wind - global_prop_wind) + abs(tst_prop_wind - global_prop_wind)
        err_torn = abs(trn_prop_torn - global_prop_torn) + abs(tst_prop_torn - global_prop_torn)
        
        total_error = err_hail + err_wind + err_torn
        
        if total_error < best_error:
            best_error = total_error
            train_dates = trn_d
            test_dates = tst_d
            
            # Early stopping if an exact match (or near exact due to floating point math) is found
            if total_error < 0.001:
                break

    print(f"Optimal split found with combined proportion error: {best_error:.4f}")
    print(f"Global Date Proportions - Hail: {global_prop_hail:.2f}, Wind: {global_prop_wind:.2f}, Torn: {global_prop_torn:.2f}")
    print(f"Train Date Proportions  - Hail: {date_groups.loc[train_dates, 'is_hail'].mean():.2f}, Wind: {date_groups.loc[train_dates, 'is_wind'].mean():.2f}, Torn: {date_groups.loc[train_dates, 'is_torn'].mean():.2f}")
    print(f"Test Date Proportions   - Hail: {date_groups.loc[test_dates, 'is_hail'].mean():.2f}, Wind: {date_groups.loc[test_dates, 'is_wind'].mean():.2f}, Torn: {date_groups.loc[test_dates, 'is_torn'].mean():.2f}")

    # Save the dates to JSON files
    train_json = json.dumps(train_dates, indent=0)
    test_json = json.dumps(test_dates, indent=0)

    ver = file_version if file_version else ""
    with open(join(BASE_PATH, f'wofs_ml_severe__training_dates{ver}.json'), "w") as file:
        file.write(train_json)
    with open(join(BASE_PATH, f'wofs_ml_severe__testing_dates{ver}.json'), "w") as file:
        file.write(test_json)

# switch to determine if baseline splitting should be performed also
baseline = False
version = "eqhaz"

train_test_splitter(baseline = baseline, test_size = 0.1, file_version = version)

print("Date based train/test splitting completed!")