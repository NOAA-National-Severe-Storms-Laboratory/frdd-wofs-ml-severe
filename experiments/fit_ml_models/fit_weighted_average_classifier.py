#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The custom classifier 
import sys, os
sys.path.insert(0, '/home/monte.flora/python_packages/wofs_ml_severe')
sys.path.insert(0, '/home/monte.flora/python_packages/ml_workflow')
sys.path.insert(0, '/home/monte.flora/python_packages/WoF_post')

from wofs.ml.load_ml_models import load_ml_model
from wofs.post.utils import load_yaml
import pandas as pd
import itertools

from ml_workflow.weighted_average_classifier import WeightedAverageClassifier
from sklearn.metrics import average_precision_score
from ml_workflow.ml_methods import brier_skill_score
from sklearn.model_selection import GroupKFold


# In[2]:


import numpy as np 
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


# ### Fit the Official Weighted Average Classifiers

# In[3]:


def return_best_val_score(model_name, target, time, 
                          resample = None, retro_str = 'realtime'): 
    """Return the best cross-validation score from the hyperparam tuning. """
    BASE_PATH =  '/work/mflora/ML_DATA/NEW_ML_MODELS/hyperopt_results'
    
    fname = os.path.join(BASE_PATH, 
                    f'{model_name}_{target}_{resample}_{time}_{retro_str}.feather')

    df = pd.read_feather(fname)
    ascending = False if model_name == "LogisticRegression" else True
    df_sorted = df.sort_values(by='loss', ascending=ascending)['loss']
    
    df_sorted.reset_index(inplace=True, drop=True)
    
    val = df_sorted[0]
    
    if val < 0:
        return -val
    else:
        return val 
    
def get_weights(model_names, time, target):
    """Compute the weights for the weighted averaging."""
    scores = [return_best_val_score(model_name, target, time) for model_name in model_names]
    total_scores = np.sum(scores)
    return scores / total_scores 
    


# In[4]:

from wofs_ml_severe.io.load_ml_models import load_ml_model

times = ['first_hour', 'second_hour']
config_path = '/home/monte.flora/python_packages/wofs_ml_severe/wofs_ml_severe/conf/ml_config_realtime.yml'
targets = ['wind_severe_0km', 'hail_severe_0km', 'tornado_severe_0km', 'all_severe', 'all_sig_severe']
model_names = ['LogisticRegression', 'XGBoost'] 

ml_config = load_yaml(config_path)
OUT_PATH = '/work/mflora/ML_DATA/NEW_ML_MODELS'  
retro_str = 'realtime'
    
for target, time in itertools.product(targets, times):
    print(f'\n Hazard: {target.upper()}....Time : {time.upper()}')
    estimators = [] 
    for model_name in model_names: 
        parameters = {
                'target' : target,
                'time' : time, 
                'drop_opt' : '',
                'model_name' : model_name,
                'ml_config' : ml_config,
            }
    
    
        model_dict = load_ml_model(**parameters)
        model = model_dict['model']
        features = model_dict['X'].columns

        estimators.append(model)

    # Fit the WeightedAverageClassifier
    weights = get_weights(model_names, time, target)
    clf = WeightedAverageClassifier(estimators,  weights=weights)
    clf.features = features
    clf.save(os.path.join(OUT_PATH, f'Average_{target}_None_{time}_{retro_str}.joblib'))
    
    print(f'Classifier Weights: {list(zip(clf.weights_, model_names))}')


# In[5]:
"""
# Check that we can successfully load the weighted average classifier. 
parameters = {
                'target' : target,
                'time' : time, 
                'drop_opt' : '',
                'model_name' : 'Average',
                'ml_config' : ml_config,
            }
    
model_dict = load_ml_model(**parameters)
model_dict['features']
model_dict['model']
"""

