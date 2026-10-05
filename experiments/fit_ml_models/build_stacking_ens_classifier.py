#======================================================
# This script builds a stacking ensemble classifier to 
# create the best combination of the tuned models trained 
# from new_train_ml_models.py. Though the actual models are
# retrained, the best performing hyperparameters are preserved.
# This allows for flexibility in training new model architectures
# and then using different combinations of models for the 
# stacking ensemble.
#
# Author: Lucas Jones (Git username : LucasJ-NSSL)
# Email: lucas.jones@noaa.gov
# Date: Oct 2nd, 2026
#======================================================

import sys
import os
import joblib
from os.path import join

# Set up paths as in your other scripts
sys.path.insert(0, '/home/lucas.jones/frdd-wofs-ml-severe')
sys.path.insert(0, '/home/lucas.jones/python_packages/frdd-ml-workflow')
sys.path.insert(0, '/home/lucas.jones/python_packages/frdd-wofs-post')

from wofs_ml_severe.common.stacking_classifier import StackingClassifier
from wofs_ml_severe.io.io import MLDataLoader
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

# Though this function is exactly reproduced from ml_trainer.py, it is used here to
# ensure the exact same CV splits are used in the stacking ensemble as the individual
# models and avoid instantiating an MLTrainer object. Ideally, the code would be less 
# redundant, so an overall utility function could be considered.
def dates_to_groups(dates, n_splits = 5): 
    """Separated different dates into a set of groups based on n_splits"""
    df = dates.copy()
    df = df.to_frame()

    unique_dates = np.unique(dates.values)

    rs = np.random.RandomState(42)
    rs.shuffle(unique_dates)

    df['groups'] = np.zeros(len(dates))
    for i, group in enumerate(np.array_split(unique_dates, n_splits)):
        df.loc[dates.isin(group), 'groups'] = i+1 
    
    groups = df.groups.values

    return groups

# the target, time, and models used in the ensemble
target = 'severe_hail'
lead_time = 'first_hour'
model_names = ['XGBClassifier', 'RFClassifier', 'LogisticRegression']
model_dir = '/work2/lucas.jones/ml_models/'
version = None      # any added identifiers to the model name used in ml_trainer.py 

n_jobs = 16

# load the training data
loader_kws = {
    'data_path' : '/work2/lucas.jones/ml_data/',
    'return_full_dataframe': False, 
    'random_state' : 123, 
    'months' : ['April', 'May', 'June'],
    'years' : [2026],  
    'mode' : 'training',
    'target_column': target,
    'lead_time': lead_time
}

print("============= STARTING STACKING ENSEMBLE PIPELINE ============='")

loader = MLDataLoader(**loader_kws)
X, y, metadata = loader.load()
groups = dates_to_groups(dates = metadata['Run Date'], n_splits = 5)
cv = list(StratifiedGroupKFold(n_splits = 5).split(X, y, groups))

# bring in the tuned models and retrain to create the stacking ensemble.
estimators = []
for model_name in model_names:

    # the file name convention used by ml_trainer.py, but additional identifiers can be added
    # in that script which can be accounted for with the version variable.
    if version is not None:
        filename = f"{model_name}_{target}_{lead_time}_rs_123_{version}.joblib"
    else:
        filename = f"{model_name}_{target}_{lead_time}_rs_123.joblib" 
    
    model_path = join(model_dir, filename)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)["model"]   # extract the model from other saved data

        if hasattr(model, "estimator"):
            model = model.estimator     # extract the estimator if calibration was performed
        
        estimators.append(model)    #extract the model information and append it

    else:
        print(f"Could not find {model_path}")

# fit the stacking classifier
print(f"Training StackingClassifier for {target} at {lead_time}...")
stacker = StackingClassifier(estimators = estimators, cv = cv, n_jobs = n_jobs)

# this clones the tuned parameters, fits them across the CV splits, 
# trains the meta-estimator, and refit the base models on the full X, y
stacker.fit(X, y)

# include the feature list explicitly for later ease in evaluation
stacker.features = list(X.columns)

if version == None:
    out_name = join(model_dir, f"StackedEnsemble_{target}_{lead_time}.joblib")
else:
    out_name = join(model_dir, f"StackedEnsemble_{target}_{lead_time}_{version}.joblib")
joblib.dump(stacker, out_name)

print("=============== STACKING ENSEMBLE COMPLETED ================")