#!/usr/bin/env python
# coding: utf-8

""" usage: stdbuf -oL python -u retrain_old_models.py  2 > & log_retrain & """

#from wofs_ml_severe.wofs_ml_severe import Classifier
from master.ml_workflow.ml_workflow.calibrated_pipeline_hyperopt_cv import CalibratedPipelineHyperOptCV

import pandas as pd
from os.path import join
import numpy as np
from ml_workflow.ml_workflow.ml_methods import norm_aupdc, brier_skill_score
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp
from glob import glob
import itertools


def scorer(model, X, y, known_skew):
    naupdc = []
    bss = [] 
    auc = []
    for n in range(10):
        inds = np.random.choice(len(X), size=len(X))
        X_i = X.iloc[inds, :]
        y_i = y[inds]
        predictions = model.predict_proba(X_i)[:,1]
        naupdc.append(norm_aupdc(y_i, predictions, known_skew=known_skew))
        bss.append(brier_skill_score(y_i, predictions))
        auc.append(roc_auc_score(y_i, predictions))
        
    print( f'NAUPDC: {np.mean(naupdc):.03f} | BSS : {np.mean(bss):.03f} | AUC : {np.mean(auc):.03f}')


def load_original_data(lead_time, target, mode='training'):
    
    target = f'matched_to_{target}_0km'
    
    df = pd.read_feather(
    f'/work/mflora/ML_DATA/DATA/original_{lead_time}_{mode}_{target}_data.feather')
    
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
    y = df[target].astype(float).values
    
    dates = df['Run Date'].apply(str)
    
    return X, y, dates

scaler = 'standard'
model_names = ['LogisticRegression', 'RandomForest']
lead_times = ['first_hour', 'second_hour']
targets = ['tornado', 'severe_hail', 'severe_wind']

for model_name, lead_time, target in itertools.product(model_names, lead_times, targets):
    # Load the original training and testing dataset. 
    target = 'tornado'
    X_train, y_train, train_dates = load_original_data(lead_time=lead_time, 
                                                       mode='training', target=target)
    
    X_test, y_test, test_dates = load_original_data(lead_time=lead_time, 
                                                    mode='testing', target=target)
    
    
    dbz_vars = [c for c in X_train.columns if 'dbz' in c and 'ens_mean' in c]
    drop_vars = [c for c in X_train.columns if 'dbz' in c and 'std' in c]
    keep_vars = [c for c in X_train.columns if c not in drop_vars]

    X_train = X_train[keep_vars]
    # Set the negative dbz values to zero. 
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    for c in dbz_vars:
        train_vals = X_train[c].values
        X_train_copy[c] = np.where(train_vals<0, 0, train_vals)
    
        test_vals = X_test[c].values
        X_test_copy[c] = np.where(test_vals<0, 0, test_vals)
    
    
    paths = glob(f'/work/mflora/ML_DATA/MODEL_SAVES/{model_name}_{lead_time}_{target}*')
    path = [p for p in paths if 'manual' not in p][0]    

    resample = None if 'None' in path else 'under'

    save_name = join('/work/mflora/ML_DATA/new_models/', 
                 f'{model_name}_{lead_time}_{target}_{resample}_{scaler}_.pkl')

    if model_name == 'LogisticRegression':
        base_estimator = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=300, random_state=42)
    else:
        base_estimator = RandomForestClassifier( n_jobs = n_jobs, criterion='entropy', random_state=42 )
    
    
    if 'Random' in model_name:
        #RandomForest Grid
        param_grid = {
               'n_estimators': hp.choice('n_estimators', [100,250,300,500, 750, 1000]),
               'max_depth': hp.choice('max_depth', [3, 5, 8, 10, 15, 20,]),
               'max_features' : hp.choice( 'max_features', [5,6,8,10, 'sqrt']),
               'min_samples_split' : hp.choice( 'min_samples_split', [4,5,8,10,15,20,25,50] ),
               'min_samples_leaf': hp.choice( 'min_samples_leaf', [4,5,8,10,15,20,25,50]),

               }   
    else:
        param_grid = {
                'l1_ratio': hp.choice('l1_ratio', [0.0001, 0.001, 0.01, 0.1, 0.5, 0.6, 0.8, 1.0]),
                'C': hp.choice('C', [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]),
                }
    
    clf = CalibratedPipelineHyperOptCV(base_estimator=base_estimator, param_grid=param_grid, 
                                   scaler=scaler, 
                                   resample=resample, n_jobs=5, max_iter=50, 
                                   cv_kwargs = {'dates': train_dates, 'n_splits': 5, 'valid_size' : 20} )

    clf.fit(X_train_copy, y_train)# params = {'C' : 0.01, 'l1_ratio' : 1.0})
    clf.save(save_name)

    # Test score
    X_test_copy = X_test_copy[X_train_copy.columns]
    scorer(clf, X_test, y_test, np.mean(y_train))

    # Train Score
    scorer(clf, X_train_copy, y_train, np.mean(y_train))

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    y_pred = clf.predict_proba(X_test_copy)[:,1]
    prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)
    sr, pod, _ = precision_recall_curve(y_test, y_pred)


    f, axes = plt.subplots(ncols=2, dpi=300, sharey=True, figsize=(8,4))

    for i, (ax, x, y) in enumerate(zip(axes.flat, [prob_pred, sr], [prob_true, pod])):
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        if i == 0:
            ax.plot([0,1], [0,1], ls='dashed')
        else:
            xx = np.linspace(0,1,100)
            yy = xx
            xx,yy = np.meshgrid(xx,xx)
            csi = 1 / (1/xx + 1/yy -1)
            ax.contourf(xx,yy,csi, cmap='Blues', alpha=0.3, levels=np.arange(0,1.1,0.1))
        
        ax.plot(x,y, color='k')

    plt.savefig(f'{model_name}_{lead_time}_{target}.png')




