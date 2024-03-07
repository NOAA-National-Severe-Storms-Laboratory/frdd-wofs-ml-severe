import re
import pandas as pd
import numpy as np
import os
from os.path import join
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor

class Concatenator():
    """Concatenator is design to concatenate all the MLDATA and MLTARGET files into
       a single dataframe. There is also processing to remove obviously bad examples 
       and correct the date issue caused by the symbolic links for 2021-2022 data. 
    """
    # Number of 5-min timesteps in an hour
    HR_SIZE = 12 
    HR_RNG = np.arange(HR_SIZE+1)
    
    TIME_RANGES = [HR_RNG, 
                  HR_RNG+HR_SIZE, 
                  HR_RNG+(2*HR_SIZE), 
                  HR_RNG+(3*HR_SIZE)]
    
    TIME_NAMES = ['first_hour', 
                 'second_hour', 
                 'third_hour', 
                 'fourth_hour']
    
    TARGET_PATTERNS = ['any', 'mesh', 'warnings']
    
    BL_PATTERN = '__prob_max'
    
    METADATA = ['forecast_time_index', 'Run Date', 'Initialization Time', 
                'obj_centroid_x', 
                'obj_centroid_y', 'label']
    
    def __call__(self, ml_files, out_path,  fix_date=True): 
    
        # Generate the full dataframe. 
        # TEMP.
        try:
            ml_files.remove('/work/mflora/SummaryFiles/20170502/0100/wofs_MLDATA_24_20170503_0230_0300.feather')
        except:
            print('Could not remove file from ml_files')
            
        try:
            ml_files.remove('/work/mflora/SummaryFiles/20180630/1800/wofs_MLDATA_45_20180630_2115_2145.feather')
        except:
            print('Could not remove file from ml_files')

        full_dataframe = self.generate_dataset(ml_files)
   
        for name, rng in tqdm(zip(self.TIME_NAMES, self.TIME_RANGES), desc='Saving datasets'): 
            # Get examples in the given time index range.
            this_df = self.lead_time_paritioning(full_dataframe, rng)
            
            if fix_date:
                this_df = self.fix_date_issue(this_df)
            
            # Save the dataset.
            this_df.to_feather(join(out_path, f'wofs_ml_severe__{name}__data.feather'))

            # Remove obviously bad data.
            this_df_fixed= self.remove_spurious_data(this_df)

            # Save the reduced dataset.
           
            this_df_fixed.to_feather(join(out_path, f'wofs_ml_severe__{name}__reduced_data.feather'))
    
    def get_numeric_init_time(self, X):
        """Convert init time (str) to number of hours after midnight. 
        WoFS extends into the next day, so hours <=12, add 24 hrs. 
        """
        Xt = X.copy()
        Xt['timestamp'] = pd.to_datetime(Xt['Initialization Time'].values, format='%H%M')
        minutes_since_midnight = lambda x: x.hour 
        Xt['Initialization Time INT'] = Xt['timestamp'].apply(minutes_since_midnight)
    
        Xt.drop(['timestamp'], axis=1, inplace=True)
    
        hrs = Xt['Initialization Time INT'].values
        Xt['Initialization Time INT'] = np.where(hrs<=12, hrs+24, hrs)
    
        return Xt 
    
    def fix_date_issue_from_symbolic_links(self, df): 
        """For symbolic linked data, the date directory changes after 0000 UTC. Need to removed
        one day from these examples
        """
        df_cp = df.copy()
    
        # Convert 'Run Date' to datetime
        df_cp['Run Date'] = pd.to_datetime(df_cp['Run Date'], format='%Y%m%d')

        # Apply the condition and subtract one day
        cond = np.where((df_cp['Run Date'].dt.year.isin([2021, 
                                                         2022])) & (df_cp['Initialization Time INT'] >= 24))[0]
        
        bad_dates = df_cp.loc[(cond), 'Run Date']
        good_dates = bad_dates - pd.to_timedelta(1, unit='d')

        #Convert back to string
        df_cp.loc[(cond), 'Run Date'] = good_dates
        
        # Convert 'Run Date' back to string in the format '%Y%m%d'
        df_cp['Run Date'] = df_cp['Run Date'].dt.strftime('%Y%m%d')
        
        df_cp.drop('Initialization Time INT', axis=1, inplace=True)
    
        return df_cp
    
    def fix_date_issue(self, df):
        # Fix date issues with the symbolic links!!
        df = self.get_numeric_init_time(df)
        df = self.fix_date_issue_from_symbolic_links(df)
        df.reset_index(inplace=True, drop=True)

        return df
        
    def concatenate_predictors_and_targets(self, ml_data_path, targets_data_path):
        """Concatenate MLDATA and MLTARGETS files into a single dataframe."""
        #print(ml_data_path, targets_data_path)
        ml_df = pd.read_feather(ml_data_path)
        targets_df = pd.read_feather(targets_data_path)

        combined_df = pd.concat([targets_df, ml_df], axis=1)

        return combined_df 
    
    def generate_dataset(self, ml_data_paths):
        """Concatenate all the dataframes into a single dataframe
        # Note to Future Monte: 
        # Unfortunately, with the data re-processing on the cloud for 2021-2022
        # the MLDATA files in 2021-2022 don't have all the extra features 
        # I computed for 2017-2020. Without access to the WRFOUT files 
        # I would be unable to re-process the data myself to add 
        # these new features. However, I do not think I should 
        # change anything. That 
        
        """
        target_data_paths = [path.replace('MLDATA', 'MLTARGETS') for path in ml_data_paths]
        which_exist = [os.path.exists(p) for p in target_data_paths]
        ml_data_paths = np.array(ml_data_paths)[which_exist]
    
        # Step 2: Define a function to read a feather file
        def read_feather(file):
            ml_df = pd.read_feather(file)
            targets_data_path = file.replace('MLDATA', 'MLTARGETS')
    
            targets_df = pd.read_feather(targets_data_path)
            combined_df = pd.concat([targets_df, ml_df], axis=1)
    
            return combined_df

        # Step 3: Parallelize reading files
        with ThreadPoolExecutor() as executor:
            dataframes = list(executor.map(read_feather, ml_data_paths))

        # Step 4: Concatenate all DataFrames
        final_df = pd.concat(dataframes, axis=0, ignore_index=True)
    
        # Deprecated. 
        #df_set = [self.concatenate_predictors_and_targets(ml_data_path, targets_data_path) for 
        #             ml_data_path, targets_data_path in zip(ml_data_paths, target_data_paths)]
        #final_df = pd.concat(df_set, axis=0, ignore_index=True)
    
        return final_df 
    
    def remove_spurious_data(self, df):
        # Remove obviously bad data. 
        feature_rngs = {'w_up__time_max__amp_ens_max_spatial_perc_90' : 90,
                'comp_dz__time_max__amp_ens_max_spatial_perc_90' : 85,
                'cape_sfc__ens_mean__spatial_mean' : 8000,
               } 
        
        df_subset = df.copy()
        for f, val in feature_rngs.items(): 
            df_subset = df_subset.loc[df_subset[f].values<=val]
        
        df_subset.reset_index(inplace=True, drop=True)

        return df_subset   
 
    def lead_time_paritioning(self, df, time_index_rng):
        # Get the examples within a particular forecast time index range.
        cond = df['forecast_time_index'].isin(time_index_rng).values
        this_df = df[cond]
        this_df.reset_index(drop=True, inplace=True)
        return this_df
    
    def determine_target_columns(self, dataframe):        
        relevant_columns = [col for col in dataframe.columns if any(re.search(pattern, col) 
                                                             for pattern in self.TARGET_PATTERNS)]
        return relevant_columns