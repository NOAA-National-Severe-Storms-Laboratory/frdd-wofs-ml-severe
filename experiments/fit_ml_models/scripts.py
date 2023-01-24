#from wofs_ml_severe.wofs_ml_severe import Classifier
from master.ml_workflow.ml_workflow.calibrated_pipeline_hyperopt_cv import CalibratedPipelineHyperOptCV

import pandas as pd
from os.path import join
import numpy as np
from ml_workflow.ml_methods import norm_aupdc, brier_skill_score
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

known_skew = {
'first_hour': 

{'severe_hail': 0.0391880873707248,
  'severe_wind': 0.027375770765324627,
  'tornado': 0.012250931885049705},

'second_hour': 

{'severe_hail': 0.03567197119293762,
  'severe_wind': 0.02379619369012823,
  'tornado': 0.009605216107852312}}


from datetime import timedelta

def correct_run_date(X):
    dt = pd.to_datetime(X['Initialization Time'].apply(str), format='%H%M')
    mask = dt.dt.strftime('%H').isin(['00', '01', '02', '03', '04', '05', '06', '07', '08', '09'])

    inds = np.where(mask==True)

    date = pd.to_datetime(X['Run Date'].apply(str), format='%Y%m%d')
    date_subset = date.iloc[inds]
    date_subset -= timedelta(days=1)
    date_subset = date_subset.dt.strftime('%Y%m%d')

    X['Run Date'].iloc[inds] = date_subset

    return X 

def scorer(model, X, y, known_skew):
    aupdc = []
    bss = [] 
    auc = []
    for n in range(30):
        inds = np.random.choice(len(X), size=len(X))
        X_i = X.iloc[inds, :]
        y_i = y[inds]
        predictions = model.predict_proba(X_i)[:,1]
        aupdc.append(average_precision_score(y_i, predictions))
        bss.append(brier_skill_score(y_i, predictions))
        auc.append(roc_auc_score(y_i, predictions))
        
    print(fr'AUPDC: {np.mean(aupdc):.03f} $\pm$ {np.std(aupdc):.03f} | BSS : {np.mean(bss):.03f} | AUC : {np.mean(auc):.03f}')
    
def load_original_data(target, mode='training', time='first_hour'):
    
    target_col = f'matched_to_{target}_0km'
    
    df = pd.read_feather(
    f'/work/mflora/ML_DATA/DATA/original_{time}_{mode}_{target_col}_data.feather')
    
    metadata = ['label', 'Run Time', 'Run Date', 'FCST_TIME_IDX']
    targets = ['matched_to_severe_hail_0km',
     'matched_to_severe_hail_15km',
     'matched_to_LSRs_0km',
     'matched_to_LSRs_15km',
     'matched_to_severe_wind_0km',
     'matched_to_severe_wind_15km',
     'matched_to_tornado_0km',
     'matched_to_tornado_15km']
    features = [f for f in df.columns if f not in targets+metadata]

    X = df[features].astype(float)
    y = df[target_col].astype(float).values
    
    dates = df['Run Date'].apply(str)
    
    return X, y, dates, df[metadata]
    
def load_ml_data(lead_time = 'first_hour', 
                 mode = 'train',
                 cols_to_drop = ['label', 'obj_centroid_x', 
                                 'obj_centroid_y', 'Run Date', 
                                 'forecast_time_index'], 
                target_col = 'hail_severe_3km_obj_match',
                ): 
    """ Loads the ML dataset. """
    # Target Var : [tornado|wind|hail]_[severe|sig_severe]_[3km, 9km, 15km, 30km]_[obj_match | ]
    #base_path = '/work/mflora/ML_DATA/MLDATA'
    #file_path = join(base_path, f'wofs_ml_severe__{lead_time}__{mode}_data.feather')
    
    base_path = '/work/mflora/ML_DATA/DATA'
    file_path = join(base_path, f'wofs_ml_severe__{lead_time}__data.feather')

    df = pd.read_feather(file_path)

    metadata = df[['Run Date', 'forecast_time_index', 'Initialization Time', 'label']]
    index = list(df.columns).index('tornado_severe_0km')
    possible_features = list(df.columns)[:index]

    drop_vars = ['QVAPOR', 'freezing_level', 
                 'stp', 'okubo_weiss', 'qv_2', 'shear_u_3to6', 'shear_v_3to6', 
                 'srh_0to500', 'Initialization Time', 'ens_max', 'cond' ]
   
    features = [f for f in possible_features if not any([d in f for d in drop_vars])]
    
    X = df[features]
    y = df[target_col]

    return X,y, metadata, df


# Convert New to Old 
def resample_to_old_dataset(X, y, old_dates):
    X_copy = X.copy()
    y_copy = y.copy()    
    new_dates = X['Run Date'].apply(str)
    
    # Get the indices of dates within the old dates 
    cond = new_dates.isin(np.unique(old_dates))
    inds = np.where(cond==True)[0]
    
    X_copy_sub = X_copy.iloc[inds]
    y_copy_sub = y_copy[inds]
    
    X_copy_sub.reset_index(drop=True, inplace=True)
    y_copy_sub.reset_index(drop=True, inplace=True)
    
    return X_copy_sub, y_copy_sub

def subset_by_year(X,y, metadata, year='2017'):
    df = X.copy()
    features = list(X.columns)
    
    df['Run Date'] = metadata['Run Date']
    df['target'] = y
    
    df = df.loc[pd.to_datetime(df['Run Date'].apply(str)).dt.strftime('%Y')==year]
    
    return df[features], df['target'].values


def subsampler(X,y, size):
    rs = np.random.RandomState(123)
    inds = rs.choice(len(X), size=size, replace=False)
    X_sub = X.iloc[inds]
    y_sub = y[inds]
    
    X_sub.reset_index(inplace=True, drop=True)
    
    return X_sub, y_sub

