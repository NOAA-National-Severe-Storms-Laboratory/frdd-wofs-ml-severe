#!/usr/bin/env python
# coding: utf-8

# ## Train the Real-Time ML Models 
# 
# The real-time ML models are trained on all available warm season cases (2017-current). 
# The following models are trained in this script: 
# 1. Severe Hail 
# 2. Severe Wind 
# 3. Tornado 
# 4. Sig. Hail
# 5. Sig. Wind 
# 6. Sig. Tornado
# 7. All-severe 
# 8. All-sig-severe 
# 
# The model classes include: 
# 1. LogisticRegression
# 2. HistGradientBoosting 
# 3. RandomForest 

# In[1]:


""" usage: stdbuf -oL python -u official_train_ml_models.py > & log_train_models & """

# In[3]:

# The custom classifier 
import sys
sys.path.append('/home/monte.flora/python_packages/wofs_ml_severe')
sys.path.insert(0, '/home/monte.flora/python_packages/ml_workflow')

from ml_workflow import TunedEstimator 
from wofs_ml_severe import load_ml_data
from wofs_ml_severe.common.emailer import Emailer 

import numpy as np

# Sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from os.path import join, exists
import os
import itertools
import multiprocessing as mp 


import sklearn.exceptions
os.environ["PYTHONPATH"] = os.path.dirname(sklearn.exceptions.__file__)
os.environ["PYTHONWARNINGS"] = "ignore::exceptions.ConvergenceWarning:sklearn.svm.base"


# In[4]:

def scorer(estimator, X, y):
    pred = estimator.predict_proba(X)[:,1]
    return 1.0 - average_precision_score(y, pred)

def dates_to_groups(dates, n_splits=5): 
    """Separated different dates into a set of groups based on n_splits"""
    df = dates.copy()
    df = df.to_frame()
    
    unique_dates = np.unique(dates.values)
    np.random.shuffle(unique_dates)

    df['groups'] = np.zeros(len(dates))
    for i, group in enumerate(np.array_split(unique_dates, n_splits)):
        df.loc[dates.isin(group), 'groups'] = i+1 
        
    groups = df.groups.values
    
    return groups

def get_search_space(model_name, X):
    if model_name == 'RandomForest':
        model = RandomForestClassifier(n_jobs=40, random_state=123)
        n_features = X.shape[1]
        search_space = {
                'criterion' : ['gini', 'entropy'],
                'n_estimators' : [100, 150, 200, 300], 
                'max_depth' : [2,5, 10, 15, 25, 40, None],
                'min_samples_split' : [2,5, 10,15,40],
                'min_samples_leaf':  [4,5,8,10,15,20,25,50],
                'max_features': list(np.arange(1, n_features)),
                'class_weight' : ['balanced', None],
                } 
        n_jobs = 1
    
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(penalty='l2', random_state=123)
        search_space = {
                'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                'class_weight' : [None, 'balanced']
                }
        n_jobs = 5
    
    elif model_name == 'XGBoost':
        n_jobs=1
        model = XGBClassifier(objective= 'binary:logistic', seed=123, 
                          tree_method='gpu_hist', gpu_id=0)
    
        search_space = {
        'n_estimators' : [50, 100, 150, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],
        'max_depth': [3,5,6,10,15,20],
        'subsample': list(np.arange(0.5, 1.0, 0.1)),
        'colsample_bytree': list(np.arange(0.5, 1.0, 0.1)),
        'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10., 100.],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10., 100.],
        'lambda': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10., 100.],
        'sampling_method' : ['uniform', 'gradient_based'],
        'min_child_weight' : list(np.arange(1, 8, 1, dtype=int)),    
    }
        
    return model, search_space, n_jobs 


def get_feature_type(X, categorical_features):
    
    # Define the categorical features for the pre-processing pipeline. 
    numeric_features = [i for i in range(len(X.columns))]
    categorical_features = [list(X.columns).index(f) for f in categorical_features]
    _ = [numeric_features.remove(i) for i in categorical_features]
    
    return categorical_features, numeric_features 


def get_target_str(target):
    # Initialize the kwargs for the hyperparameter optimization.
    if isinstance(target, list):
        if 'sig_severe' in target[0]:
            target = 'all_sig_severe'
        else:
            target = 'all_severe'
   
    return target 
        

# In[5]:


OUT_PATH = '/work/mflora/ML_DATA/NEW_ML_MODELS'

#  ['wind_severe_0km', 'hail_severe_0km', 'tornado_severe_0km'],
# ['wind_sig_severe_0km', 'hail_sig_severe_0km', 'tornado_sig_severe_0km']

# Notes:
# For the first round of experiments, recreate the models from the 2021 paper. 
target_cols = ['wind_severe_0km', 'hail_severe_0km', 'tornado_severe_0km', 
               'wind_sig_severe_0km', 'hail_sig_severe_0km', 'tornado_sig_severe_0km',
              ]
model_names = ['XGBoost', 'LogisticRegression', 'RandomForest']
resampling = [None]
times = ['first_hour', 'second_hour', 'third_hour', 'fourth_hour']


# If 'training', it will load the data with the original training dates from Flora et al. (2021, MWR)
# If None, train the operational models. 
modes = [None]

# If True, existing files will be overwritten; setting to False is useful when the scripts
# fails to finish and I don't want to redo stuff. 
overwrite = False

emailer = Emailer()

for target, model_name, time, resample, mode in itertools.product(target_cols, 
                                                            model_names,
                                                            times,
                                                            resampling, 
                                                                  modes): 
   
    def fitting(): 
        retro_str = 'retro' if mode == 'training' else 'realtime'
    
        target_str = get_target_str(target)
    
        fname = join(OUT_PATH, 
                 f'{model_name}_{target_str}_{resample}_{time}_{retro_str}.joblib')
    
        if not overwrite:
            if exists(fname):
                return None 
            
        if target_str in ['all_severe', 'all_sig_severe'] and retro_str =='retro':
            #continue 
            return None
            
    
        print('\nTraining a new model....')
        subject = f"""Target: {target} 
          Model Name : {model_name} 
          Lead Time: {time} 
          Resample: {resample} 
          Mode: {retro_str}\n"""
        print(subject)
    
        start_time = emailer.get_start_time()

        # Load the data. Using the run dates, we can group the data into 
        # 5 cross-validation folds, which will be used for hyperparameter optimization
        # and training the calibration model. 
        X, y, metadata = load_ml_data(target_col=target, 
                                  lead_time=time,
                                  mode=mode,
                                 )
    
        dates = metadata['Run Date']
        groups = dates_to_groups(dates, n_splits=5)
    
        model, search_space, n_jobs = get_search_space(model_name, X)
    
        categorical_features, numeric_features = get_feature_type(X, categorical_features=['Initialization Time'])
    
        # Initialize the cross-validation groups 
        cv = list(GroupKFold(n_splits=5).split(X,y,groups))
    
        output_fname = join(OUT_PATH, 'hyperopt_results', 
                        f'{model_name}_{target_str}_{resample}_{time}_{retro_str}.feather')
    
        hyperopt_kwargs = {'search_space' : search_space, 
                   'optimizer' : 'tpe', 
                   'max_evals' : 100, 
                   'patience' : 50, 
                  'scorer' : scorer, 
                  'n_jobs' : n_jobs, 
                  'cv' : cv, 
                  'output_fname' : output_fname 
                      }
    
        # Initialize the kwargs for the Pipeline. 
        pipeline_kwargs={'imputer' : 'simple', 
                     'resample': resample, 
                     'scaler': 'standard', 
                     'numeric_features' : numeric_features, 
                     'categorical_features' :  categorical_features}
    
        # Initialize the kwargs for the calibration model. 
        calibration_cv_kwargs = {'method' : 'isotonic', 'ensemble' : False, 
                         'cv' : cv, 'n_jobs': n_jobs}

        # Fit the model and save it. 
        estimator = TunedEstimator(model, pipeline_kwargs, hyperopt_kwargs, calibration_cv_kwargs)
    
        if hasattr(y, 'values'):
            y = y.values
    
        estimator.fit(X,y,groups)
        estimator.save(fname)
        del estimator
        #save the model here on the disk

        try:
            emailer.send_email(subject, start_time)
        except:
            print('Unable to send email. Possibly an NSSL network issue.')

    fitting_process = mp.Process(target=fitting)
    fitting_process.start()
    fitting_process.join()

