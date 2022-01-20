import numpy as np 
from hyperopt import hp

def get_param_grid( model_name ):
    if 'Random' in model_name:
             #RandomForest Grid
        param_grid = {
               'n_estimators': hp.choice('n_estimators', [100,250,300,500, 750, 1000]),
               'max_depth': hp.choice('max_depth', [3, 5, 8, 10, 15, 20,]),
               'max_features' : hp.choice( 'max_features', [5,6,8,10, 'sqrt']),
               'min_samples_split' : hp.choice( 'min_samples_split', [4,5,8,10,15,20,25,50] ),
               'min_samples_leaf': hp.choice( 'min_samples_leaf', [4,5,8,10,15,20,25,50]),

               }
    elif 'XGB' in model_name:
        param_grid = {
                   'n_estimators' : hp.choice('n_estimators', np.arange(100, 1000, 10, dtype=int)),
                   'gamma' : hp.quniform('gamma', 0.2, 1, 0.05), 
                   'max_depth': hp.choice('max_depth', np.arange(3, 16, 1, dtype=int)),
                   'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
                   'min_child_weight' : hp.choice('min_child_weight', np.arange(1, 20, 1, dtype=int)),
                   'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
                   'subsample': hp.uniform('subsample', 0.6, 0.9),
                   'reg_alpha': hp.choice( 'reg_alpha', [0, 0.25, 0.5, 1, 10, 15]),
                   'reg_lambda': hp.choice( 'reg_lambda', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0]),
                }

    elif 'Logistic' in model_name:
        param_grid = {
                'l1_ratio': hp.choice('l1_ratio', [0.0001, 0.001, 0.01, 0.1, 0.5, 0.6, 0.8, 1.0]),
                'C': hp.choice('C', [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]),
                }

    elif 'Network' in model_name:
        param_grid = {
                'choice': hp.choice('layers_number',
                             [  {'layers': 'two'},
                                {'layers': 'three',
                                 'units3': hp.choice('units3', np.arange(8,256,32)),
                             }]),
                'units1': hp.choice('units1', np.arange(32, 1024,32, dtype=int)),
                'units2': hp.choice('units2', np.arange(32, 512,32, dtype=int)),
                'l2_weight' : hp.uniform('l2_weight', 0.001, 0.1),
                'dropout_fraction' : hp.uniform('dropout_fraction', 0.25, 0.75),
                'use_batch_normalization' : hp.choice('use_batch_normalization', [True, False]),
                'batch_size' : hp.choice('batch_size', [128, 256, 512, 1024]),
                'n_epochs' :  hp.choice('n_epochs', [30, 50, 100]),
                }
    else:
        raise ValueError(f'{model_name} is an accepted options!')


    return param_grid



