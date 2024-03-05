import pandas as pd
from os.path import join, exists, dirname, realpath
import numpy as np
from sklearn.model_selection import train_test_split
import random
import json

from importlib_resources import files

def get_numeric_init_time(X):
        """Convert init time (str) to number of hours after midnight. 
        WoFS extends into the next day, so hours <=12, add 24 hrs. 
        """
        Xt = X.copy()
        Xt['timestamp'] = pd.to_datetime(Xt['Initialization Time'].values, format='%H%M')
        minutes_since_midnight = lambda x: x.hour 
        Xt['Initialization Time'] = Xt['timestamp'].apply(minutes_since_midnight)
    
        Xt.drop(['timestamp'], axis=1, inplace=True)
    
        hrs = Xt['Initialization Time'].values
        Xt['Initialization Time'] = np.where(hrs<=12, hrs+24, hrs)
    
        return Xt 

class MLDataLoader:
    """ Loads the ML dataset. 
    
    Attributes
    ---------------------
    
    target_colum : str or list 
        The target column. For the official models, the target column is a alias rather a specific
        column in the dataframe. If a list, then the different columns are summed together 
        and re-binarized; this is useful for creating all-severe or all-sig-severe targets.
        
    lead_time : 'first_hour', 'second_hour', 'third_hour', or 'fourth_hour' or list thereof
        The lead time range. If a list, multiple dataframes are loaded and concatenated
        
    mode : 'training', 'testing', 'total' or None (default is None).
        If 'training' or 'testing', the dataset is loaded with the training or 
        testing datasets. If None, loads the training datasets. If 'total', 
        loads the full dataset. 
        
    return_full_dataframe: bool (default=True)
        If True, returns the full dataframe prior to separating into X,y
    
    load_reduced_dataframe : bool (default=False)
        If True, returns the "reduced" dataset where data cleaning has been applied 
    
    load_baseline_dataframe : bool (default=False)
        If True, returns the dataframe with the baseline features 
    
    test_size : float (default=0.3)
        The portion of the dates used for the testing dataset 
        
    data_path : str or path-like
        Path to the parent dataframe files 
        
    random_state : int or None 
        Random state integer for the training/testing split 
        
    excluded_years : list of int
        List of the years to include from the dataset 
    
    exclude_missing_mesh : bool (default=True)
        If True, removing examples where the mesh == -1 (missing) 
        
    alter_init_times : bool (default=False)
        If True, convert the string init time values to categories.
        
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
    SPATIAL_TXT = 'spatial_mean'
    AMP_TXT = 'amp_ens'
    
    OBJECT_FEATURES = ['area','eccentricity', 'extent', 'orientation',
                       'minor_axis_length', 'major_axis_length', 'ens_track_prob',
                       'area_ratio']
    
    METADATA_FEATURES = ['Run Date', 'forecast_time_index', 
                         'Initialization Time', 'label', 
                         'obj_centroid_y', 'obj_centroid_x']
    
    SEVERE_HAIL_THRESH = 1.0
    SEVERE_WIND_THRESH = 50.0
    SEVERE_TORN_THRESH = 1
    
    SIG_SEVERE_HAIL_THRESH = 2.0
    SIG_SEVERE_WIND_THRESH = 65.0
    SIG_SEVERE_TORN_THRESH = 2
    
    # These are the init times to keep. It ignores init times from 1700-1900 (not inclusive). 
    # These are the standard init times used during the spring cases. 
    INIT_TIMES = ['0000', '0030', '0100', '0130', 
                  '0200', '0230', '0300', '1900', 
                  '1930', '2000', '2030', '2100',
                  '2130', '2200', '2230', '2300', 
                  '2330']
    
    # Hours after midnight. 
    #INIT_TIMES = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23]
    
    def __init__(self, target_column=None, lead_time='first_hour',  mode='training', 
                 return_full_dataframe = False, load_reduced_dataframe=True, 
                 load_baseline_dataframe=False, 
                 data_path = '/work/mflora/ML_DATA/DATA', 
                 random_state=123, 
                 months = ['April', 'May', 'June'],
                 years = [2018, 2019, 2020, 2021, 2022], 
                 exclude_missing_mesh=False, 
                 alter_init_times = False, load_multiple_y=False,
                 drop_features = True, 
                ):
        
        self._load_baseline = load_baseline_dataframe
        self._load_reduced = load_reduced_dataframe
        self.return_full_dataframe = return_full_dataframe
        self.exclude_missing_mesh = exclude_missing_mesh
        self.years = years
        self.months = months
        
        self.target_column = target_column
        self.lead_time = lead_time
        self.mode = self.get_mode(mode) 
        self.random_state = random_state

        self.data_path = data_path
        self.alter_init_times = alter_init_times
        self.load_multiple_y=load_multiple_y
        self.drop_features = drop_features
    
    def load(self):
        
        dataframe = self._load_dataframe()
        if self.return_full_dataframe:
            return dataframe 
        
        # Split data into training or testing based on the given mode.
        dataframe = self.split_data(dataframe)
        
        # Sample by init times
        dataframe = self.sample_by_init_time(dataframe)
        
        # Drop data where mesh is missing. 
        dataframe = self.sample_by_missing_mesh(dataframe)
        
        # (optional) convert init times to category 
        dataframe = self.convert_init_times_to_category(dataframe)
        
        # Convert NaNs to zeros.
        dataframe = self.inf_to_nan(dataframe)
        
        X = self.get_X(dataframe)
        
        # Drop out features. 
        X = self.remove_features(X)
        
        # check that the subsampling worked!
        assert len(X) > 10000, 'X is too small! Issue with the subsampling'
        
        if isinstance(self.target_column, str):
            self.load_multiple_y = False
        
        if self.load_multiple_y:
            y = [self.get_y(dataframe, t) for t in self.target_column]
        else:
            y = self.get_y(dataframe, self.target_column)
        metadata = self.get_metadata(dataframe)
        
        return X,y, metadata 
    
    def inf_to_nan(self, dataframe):
        """Convert NaNs to zeros"""
        # For the conditional features, replace the NaNs with zero
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.reset_index(inplace=True, drop=True) 
    
        return dataframe
    
    def convert_init_times_to_category(self, dataframe):
        """Convert the str init times to hours after midnight"""
        # Convert the str init times to hours after midnight.
        if self.alter_init_times:
            dataframe = get_numeric_init_time(dataframe)
            
        return dataframe
    
    def get_numeric_init_time(self, X):
        """Convert init time (str) to number of hours after midnight. 
        WoFS extends into the next day, so hours <=12, add 24 hrs. 
        """
        Xt = X.copy()
        Xt['timestamp'] = pd.to_datetime(Xt['Initialization Time'].values, format='%H%M')
        minutes_since_midnight = lambda x: x.hour 
        Xt['Initialization Time'] = Xt['timestamp'].apply(minutes_since_midnight)
    
        Xt.drop(['timestamp'], axis=1, inplace=True)
    
        hrs = Xt['Initialization Time'].values
        Xt['Initialization Time'] = np.where(hrs<=12, hrs+24, hrs)
    
        return Xt 
    
    def sample_by_missing_mesh(self, dataframe):
        """Drop data where the MESH is missing (values of -1)"""
         # Drop data where the MESH is missing.
        if self.exclude_missing_mesh:
            dataframe = dataframe.loc[(dataframe['mesh_severe_0km']>=0) & (dataframe['max_mesh']>=0) ] 
            dataframe.reset_index(inplace=True, drop=True) 
            
            dataframe = dataframe.loc[(dataframe['max_mesh']<=10) ] 
            dataframe.reset_index(inplace=True, drop=True) 
            
    
        return dataframe
    
    def sample_by_init_time(self, dataframe):
        """Sample by init time and only keep init times after the initial WoFS cycling"""
        dataframe = dataframe.loc[dataframe['Initialization Time'].isin(self.INIT_TIMES)]
        dataframe.reset_index(inplace=True, drop=True) 
    
        return dataframe
    def _train_test_split(self, dataframe, return_dates=False):
        """Code for the train/test splitting. """
        months_str = f"months:{'_'.join(self.months)}"
        years = [str(y) for y in self.years]
        years_str = f"years:{'_'.join(years)}"
        fname = f"train_test_case_split_{years_str}_{months_str}_rs:{self.random_state}.json"
    
        # Get the directory of the current file (my_module.py)
        dir_path = dirname(realpath(__file__))
    
        fname = join(dir_path, fname)
        
        if not exists(fname):
            raise FileNotFoundError(f"""{fname} not found! 
                                    Check that this file exist for the 
                                    given years : {self.years}, months : {self.months}, and 
                                    random state : {self.random_state}. If not, create with the 
                                    'create_date_based_train_test_split.ipynb' in the fit_ml_models dir""")
            
        with open(fname, "r") as file:
            date_dict = json.load(file)
            
        return date_dict
    
    def split_data(self, dataframe):
        """Perform the training or testing split of the data. 
        The train/test split determined randomly by WoFS run date."""
        
        cases_split = self._train_test_split(dataframe, True)
 
        these_dates = cases_split[f'{self.mode}_dates']
        dataframe = self.resample_by_date(dataframe, these_dates)
        
        return dataframe
        
    def resample_by_date(self, dataframe, dates):
        """Using a list of dates, resample the dataframe"""
        existing_dates = dataframe['Run Date'].apply(str)
        dataframe_rs = dataframe.loc[existing_dates.isin(dates)]
        dataframe_rs.reset_index(inplace=True, drop=True)
    
        return dataframe_rs
    
    
    def _nn_train_test_split(self, X, y, metadata, test_size=0.25, random_state=1234):
        """Split the training dataset into training and 
        validation based on date. Used for training neural network models 
        for early stopping. 
        """
        def to_list_of_str(a):
            return list(a.astype(str))
        
        features = X.columns
        
        dataframe = pd.concat([X, metadata], axis=1)
        dataframe['target'] = y
        
        # Split the dates into train and validatoin
        unique_run_dates = np.unique(dataframe['Run Date'])
        train_dates, val_dates = train_test_split(unique_run_dates, 
                                                test_size=test_size, 
                                                random_state=random_state)
        
        # Check that training and testing dates do not overlap!
        if not set(train_dates).isdisjoint(set(val_dates)):
            raise ValueError("Overlap found between training and testing dates.")
        
        print(f"Num of Train Dates: {len(train_dates)}...Num. of Val. Dates: {len(val_dates)}")
        
        train_dates = to_list_of_str(train_dates)
        val_dates = to_list_of_str(val_dates)
        
        train_df = self.resample_by_date(dataframe, train_dates)
        val_df = self.resample_by_date(dataframe, val_dates)
        
        X_train, y_train = train_df[features], train_df['target']
        X_val, y_val = val_df[features], val_df['target']
        
        return X_train, X_val, y_train, y_val
    
    
    def get_mode(self, mode):
        mode = 'training' if mode is None else mode
        return mode
    
    def _load_dataframe(self):
        """Load and possibly concatenate feather dataframe files based on lead times."""
        if isinstance(self.lead_time, list):
            # Handle the case where lead_time is a list
            dataframes = []
            for lead_time in self.lead_time:
                file_path = join(self.data_path, f'wofs_ml_severe__{lead_time}__data.feather')
            
                if self._load_baseline:
                    file_path = file_path.replace('data', 'baseline_data')
                
                if self._load_reduced:
                    file_path = file_path.replace('data', 'reduced_data')

                df = pd.read_feather(file_path)
                dataframes.append(df)

            # Concatenate all dataframes along the row axis
            dataframe = pd.concat(dataframes, axis=0)
            dataframe.reset_index(inplace=True, drop=True)
            
            return dataframe
            
        else:
            # Original functionality for single lead_time
            file_path = join(self.data_path, f'wofs_ml_severe__{self.lead_time}__data.feather')
        
            if self._load_baseline:
                file_path = file_path.replace('data', 'baseline_data')
            
            if self._load_reduced:
                file_path = file_path.replace('data', 'reduced_data')

            return pd.read_feather(file_path)
        
    def get_metadata(self, dataframe):
        """Get the meta data features like init time, run date, etc."""
        return dataframe[self.METADATA_FEATURES]
    
    def get_X(self, dataframe):
        """Using the feature columns, get the X from the parent dataframe"""
        all_columns = list(dataframe.columns)
        
        spatial_features = [f for f in all_columns if self.SPATIAL_TXT in f]
        amp_features = [f for f in all_columns if self.AMP_TXT in f]
    
        features = spatial_features + amp_features + self.OBJECT_FEATURES 
        
        return dataframe[features]
    
    def get_y(self, dataframe, target_column):
        """Using the given target column, get the y from the parent dataframe"""
    
        def compute_binary_target(column, threshold=0):
            return np.where(dataframe[column].values >= threshold, 1, 0)

        def compute_sum_target(columns, thresholds):
            conditions = [compute_binary_target(col, thresh) for col, thresh in zip(columns, thresholds)]
            return np.where(np.sum(conditions, axis=0) > 0, 1, 0)

        target_mappings = {
            'severe_hail': lambda df: compute_binary_target('hail_any_0km', self.SEVERE_HAIL_THRESH),
            'severe_wind': lambda df: compute_binary_target('wind_any_0km', self.SEVERE_WIND_THRESH),
            'severe_torn': lambda df: compute_binary_target('tornado_any_0km', self.SEVERE_TORN_THRESH),
            'severe_mesh': lambda df: df['mesh_severe_30mm_0km'].values,
            'sig_severe_hail': lambda df: compute_binary_target('hail_any_0km', self.SIG_SEVERE_HAIL_THRESH),
            'sig_severe_wind': lambda df: compute_binary_target('wind_any_0km', self.SIG_SEVERE_WIND_THRESH),
            'any_severe': lambda df: compute_sum_target(['hail_any_0km', 'wind_any_0km', 'tornado_any_0km'],
                                                    [self.SEVERE_HAIL_THRESH, self.SEVERE_WIND_THRESH, 
                                                     self.SEVERE_TORN_THRESH]),
            'any_sig_severe': lambda df: compute_sum_target(['hail_any_0km', 'wind_any_0km', 'tornado_any_0km'],
                                                        [self.SIG_SEVERE_HAIL_THRESH, 
                                                         self.SIG_SEVERE_WIND_THRESH, 
                                                         self.SIG_SEVERE_TORN_THRESH]),
            'severe_warn': lambda df: df['severe_weather_warnings_0km'].values,
            'torn_warn': lambda df: df['tornado_warnings_0km'].values,
            'hail': lambda df: df['hail_any_0km'].values,
            'wind': lambda df: df['wind_any_0km'].values,
            'mesh': lambda df: df['max_mesh'].values,
            'tornado_probsevere': lambda df: get_tornado_probsevere(df)['tornado_probsevere']
        }

        if isinstance(target_column, list):
            ts = np.zeros((len(dataframe), len(target_column)), dtype=int)
            for i, t in enumerate(target_column):
                ts[:,i] = target_mappings.get(t, lambda df: df[t].values)(dataframe)
            return np.where(np.sum(ts, axis=1) > 0, 1, 0)

        else:
            return target_mappings.get(target_column, lambda df: df[target_column])(dataframe)
    
    # Possible Deprecated!    
    def get_tornado_probsevere(self, df):
        """
        Using the logic from ProbSevere, the tornado targets are based on 
        1 if tornado is observed and 0 is hail or wind is observed, but no 
        tornado. 
        """
        targets = df[['tornado_severe_0km', 'wind_severe_0km', 'hail_severe_0km']]

        # create new column with desired logic
        df['tornado_probsevere'] = df.apply(lambda row: 1 if row['tornado_severe_0km'] == 1 else 
                        (0 if (row['hail_severe_0km'] == 1 or row['wind_severe_0km'] == 1) else np.nan), axis=1)

        df.dropna(subset=['tornado_probsevere'], inplace=True)
    
        return df     
    
    def remove_features(self, X, cc_val=0.85): 
        """Drop features from the input"""
        drop_vars = self.OBJECT_FEATURES
    
        # Removing any of the new features. 
        removed_features = ['QVAPOR_850', 
                            'QVAPOR_700', 
                            'QVAPOR_500', 
                            'qv_2', 
                            'freezing_level', 
                            'srh_0to500',
                            'shear_u_3to6', 'shear_v_3to6', 
                            'stp', 'stp_srh0to500', 'dbz_1km',
                            'okubo_weiss',
                            'temperature_500',
                            'temperature_700',
                   'temperature_850',
                   'avg_updraft_track_area',
                   'td_700',
                   'td_850',
                   'td_500',
                   'geo_hgt_850',
                   'geo_hgt_700',
                   'geo_hgt_500',
                   'theta_e', 
                  ] 
    
        for f in X.columns:
            # No longer using the conditional features. 
            if 'cond' in f:
                drop_vars.append(f) 
            # The new amplitude features specify the spatial percentile (10, 90)
            if 'amp' in f and 'perc' not in f:
                drop_vars.append(f)
            #if 'ens_max' in f:
            #    drop_vars.append(f)
            if any([v in f for v in removed_features]):
                drop_vars.append(f)
            # Not longer using the instanteous version of UH and vert vort.
            if any([n in f for n in ['uh', 'wz']]) and 'instant' not in f:
                drop_vars.append(f)
                
            # The mixed layer computations changed in 2023 to fix a known bug. Unfortunately, 
            # since we cannot reprocess the prior summary files. We have to remove these features.
            if any([v in f for v in ['cape_ml', 'cin_ml', 'lcl_ml', 'cape_mu', 'cin_mu', 'lcl_mu']]):
                drop_vars.append(f)
            
        X.drop(drop_vars, axis=1, inplace=True)
    
        #corr_filter = CorrelationFilter()
        #ens_std_vars = [f for f in X.columns if 'ens_std' in f]
        #X_keep = X[ens_std_vars]
        #X.drop(ens_std_vars, axis=1, inplace=True)
        #X, dropped_columns, corr = corr_filter.filter_dataframe(X, cc_val)
        # Add the ens std vars back in. 
        #X = pd.concat([X, X_keep], axis=1)
    
        return X 
    

class HailSizeLoader:
    def __init__(self, lead_time, mode=None):
        self.lead_time = lead_time 
        self.mode = mode

    def load(self, mode='training'):
        loader_kws= {
             'data_path' :  '/work/mflora/ML_DATA/DATA/',
              'return_full_dataframe': False, 
              # Random state is fixed, so the data is compatible 
              # with the classification models. 
              'random_state' : 123, 
              'mode' : mode,
              'exclude_missing_mesh': True,
              'target_column' : ['mesh', 'hail'],
              'lead_time' : self.lead_time,
              'load_multiple_y' : True
                            }
        
        # Load the training dataset
        loader = MLDataLoader(**loader_kws)
        X, y, metadata = loader.load()

        # Get the combined MESH and LSR hail sizes. 
        y = self.get_combined_hail(y)

        if self.mode == 'semi_supervised':
            # Limit data samples to those where the HAILCAST > 0 (ignore misses at the moment)
            X, y, metadata = self.resample_by_hailcast(X, y, metadata, cond=0.75)
        
        elif self.mode:
            if self.mode == 'small_hail':
                # Limit data samples to those where the target is not more than 0.5 in greater than 
                # the hailcast prediction. That is to limit the misses on lowering the prediction. 
                
                #X_train, y_train, metadata = self.resample_by_hailcast(X_train, y_train, 
                #                                                       metadata, cond=0.5)
                
                X, y, metadata = self.hail_sampler(X, y, metadata, rng=[0.0, 2.0])
                
            elif self.mode == 'large_hail':
                X, y, metadata = self.hail_sampler(X, y, metadata, rng=[0.5, 10])
                
            elif self.mode == 'massive_hail':
                X, y, metadata = self.hail_sampler(X, y, metadata, rng=[0.5, 10])
        
        
        # Split into training and validation 
        if mode == 'training':
            X_train, X_val, y_train, y_val = loader._nn_train_test_split(X, y, metadata)
        
            self.check_for_nans_and_inf(X_train, y_train, X_val, y_val)
            
            return X_train, X_val, y_train, y_val
        
        else:
            return X, y, metadata
        
    
    def check_for_nans_and_inf(self, X_train, y_train, X_val, y_val ):
        print(np.max(y_train), np.max(y_val))
        
        assert np.min(y_train) >= 0, 'Negative values in y_train!'
        assert np.min(y_val) >= 0, 'Negative values in y_val!'
        
        assert not np.any(np.isinf(X_val)), 'Infinite values in X_val!'
        assert not np.any(np.isinf(y_val)), 'Infinite values in y_val!'
        assert not np.any(np.isnan(X_val)), 'NaN values in X_val!'
        assert not np.any(np.isnan(y_val)), 'NaN values in y_val!'
        
        print(f"{np.any(np.isnan(X_val))=}")
        print(f"{np.any(np.isnan(y_val))=}")
        print(f"{np.any(np.isinf(X_val))=}")
        print(f"{np.any(np.isinf(y_val))=}")
           
    def hail_sampler(self, X,y, metadata, rng=[0, 0.75]): 
        """Function subsamples the data based on the provided hail size range."""
    
        features = X.columns
        metadata_cols = metadata.columns
        
        # Assuming X and y are pandas DataFrames or Series
        # Concatenate X and metadata
        dataframe = pd.concat([X, metadata], axis=1)
        dataframe['target'] = y
    
        # Filter rows based on the range
        dataframe = dataframe[(dataframe['target'] >= rng[0]) & (dataframe['target'] <= rng[1])]
    
        # Reset index after filtering
        dataframe.reset_index(inplace=True, drop=True)
    
        # Assuming the need to separate the features and target again
        X_rs = dataframe[features]
        y_rs = dataframe['target'].values
    
        return X_rs, y_rs, dataframe[metadata_cols]

    def get_combined_hail(self, y):
        """Function used to combine MESH and LSR hail sizes into a single target variable. 
           1. Where LSR size > 0, add noise of N(0.05, 0.05)
           2. If MESH == 0, keep 
           3. If LSR == 0, but MESH > 0, keep MESH
           4. If LSR and MESH > 0, weighted avg of both, weighted towards the LSR
              if MESH < 2 otherwise keep LSR
        """
        # Function to combine hail and mesh values based on the provided conditions
        def combine_hail_mesh(row):
            if row['mesh'] == 0:
                return row['lsr']
            elif row['lsr'] == 0 and row['mesh'] != 0:
                return row['mesh']
            elif row['lsr'] != 0 and row['mesh'] != 0:
                if row['mesh'] < 2:
                    # Weighted towards the LSR
                    return 0.75*row['lsr'] + 0.25*row['mesh']
                else:
                    return row['lsr']
            else:
                return None  # This case should not occur based on the given conditions
    
        # Example DataFrame
        data = {
            'lsr': y[1],
            'mesh': y[0]
        }
        df = pd.DataFrame(data)

        df['original_lsr'] = df['lsr'].copy()

        # Identify the indices where 'lsr' is greater than zero
        indices = df['lsr'] > 0

        # Generate random noise for these indices
        noise = np.random.normal(0.05, 0.05, sum(indices))

        # Add noise only to those elements in 'lsr' where the condition is True
        df.loc[indices, 'lsr'] += noise

        # Clip values to be non-negative
        df['lsr'] = np.clip(df['lsr'], 0, None)

        # Apply the function to each row
        df['combined'] = df.apply(combine_hail_mesh, axis=1)

        return df['combined'].values
    
    
    
    
#####################################################################################    
# LEGACY CODE......................
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
        
def my_train_test_split():

    lead_time = 'first_hour'
    base_path = '/work/mflora/ML_DATA/DATA'
    file_path = join(base_path, f'wofs_ml_severe__{lead_time}__reduced_data.feather')
    df = pd.read_feather(file_path)

    # Get the warm season cases!
    df = df.loc[pd.to_datetime(
            df['Run Date'].apply(str)).dt.strftime('%B').isin(['April', 'May', 'June'])]

    run_dates = df['Run Date']

    unique_run_dates = np.unique(run_dates)

    train_dates, test_dates = train_test_split(unique_run_dates, test_size=0.3, random_state=123)
    train_dates = to_list_of_str(train_dates)
    test_dates = to_list_of_str(test_dates)
    
    data = {'testing_dates' : test_dates, 'training_dates' : train_dates}
    with open("/work/mflora/ML_DATA/DATA/train_test_case_split.json", "w") as outfile:
        json.dump(data, outfile)
        
# Convert New to Old 
def resample_to_old_dataset(df, original_dates):
    dates = df['Run Date'].apply(int)
    return df.loc[dates.isin(original_dates)]


def resample_by_date(df, dates):
    existing_dates = df['Run Date'].apply(str)
    df_rs = df.loc[existing_dates.isin(dates)]
    
    df_rs.reset_index(inplace=True, drop=True)
    
    return df_rs

def get_numeric_init_time(X):
    """Convert init time (str) to number of hours after midnight. 
    WoFS extends into the next day, so hours <=12, add 24 hrs. 
    """
    Xt = X.copy()
    Xt['timestamp'] = pd.to_datetime(Xt['Initialization Time'].values, format='%H%M')
    minutes_since_midnight = lambda x: x.hour 
    Xt['Initialization Time'] = Xt['timestamp'].apply(minutes_since_midnight)
    
    Xt.drop(['timestamp'], axis=1, inplace=True)
    
    hrs = Xt['Initialization Time'].values
    Xt['Initialization Time'] = np.where(hrs<=12, hrs+24, hrs)
    
    return Xt 


def get_tornado_probsevere(df):
    """
    Using the logic from ProbSevere, the tornado targets are based on 
    1 if tornado is observed and 0 is hail or wind is observed, but no 
    tornado. 
    """
    targets = df[['tornado_severe_0km', 'wind_severe_0km', 'hail_severe_0km']]

    # create new column with desired logic
    df['tornado_probsevere'] = df.apply(lambda row: 1 if row['tornado_severe_0km'] == 1 else 
                        (0 if (row['hail_severe_0km'] == 1 or row['wind_severe_0km'] == 1) else np.nan), axis=1)

    df.dropna(subset=['tornado_probsevere'], inplace=True)
    
    return df 

def load_ml_data(target_col, 
                 lead_time = 'first_hour', 
                 mode = None, 
                 baseline=False,
                 return_only_df=False, 
                 load_reduced=True, 
                 base_path = '/work/mflora/ML_DATA/DATA',
                 alter_init_times=True,
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
    
    #TRAIN_YEARS = ['2018', '2019', '2020']
    #TEST_YEARS = ['2017', '2021']
    
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
 
    if target_col == 'tornado_probsevere':
        df = get_tornado_probsevere(df)

    # Deprecated!!!
    # Get the warm season cases!
    #df = df.loc[pd.to_datetime(
    #        df['Run Date'].apply(str)).dt.strftime('%B').isin(['April', 'May', 'June'])]
    # Split into training or testing based on year. 
    # Default the training dataset is loaded. 
    #years = TRAIN_YEARS if mode in ['training', None] else TEST_YEARS
    #df = df.loc[pd.to_datetime(
    #        df['Run Date'].apply(str)).dt.strftime('%Y').isin(years)]
    
    mode = 'training' if mode is None else mode
    
    with open("/work/mflora/ML_DATA/DATA/train_test_case_split.json", "r") as outfile:
        cases_split = json.load(outfile)
    
    these_dates = cases_split[f'{mode}_dates']
    df = resample_by_date(df, these_dates)
    
    # These are the init times to keep. It ignores init times from 1700-1900 (not inclusive). 
    # These are the standard init times used during the spring cases. 
    init_times = ['0000', '0030', '0100', '0130', '0200', '0230', '0300', '1900', '1930', '2000', '2030', '2100',
       '2130', '2200', '2230', '2300', '2330']

    df = df.loc[df['Initialization Time'].isin(init_times)]
    df.reset_index(inplace=True, drop=True) 
    
    metadata_features = ['Run Date', 'forecast_time_index', 
                         'Initialization Time', 'label', 'obj_centroid_y', 'obj_centroid_x']
    metadata = df[metadata_features]
    
    # Convert the str init times to hours after midnight.
    if alter_init_times:
        df = get_numeric_init_time(df)
    
    # For the conditional features, replace the NaNs with zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.reset_index(inplace=True, drop=True) 
    
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
        
  