# Python imports
from os.path import join 

# Intel-based scikit-learn packages.
# Improves the speed! 
from sklearnex import patch_sklearn
patch_sklearn()


# Third-party imports.
import numpy as np 
from hyperopt import hp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline


# From this repository 
from ..common.util import Emailer
from ..io.io import IO 

# From ml_workflow
import sys
sys.path.append('/home/monte.flora/python_packages/ml_workflow')
from ml_workflow import CalibratedHyperOptCV
from ml_workflow import PreProcessPipeline

class Classifier(Emailer):
    def __init__(self, n_jobs=25):
        self.ML_MODEL_SAVE_PATH = '/work/mflora/ML_DATA/NEW_ML_MODELS'
        self._n_jobs=n_jobs
        
    def fit(self, model_name, X,y, dates, categorical_features=None, params=None):
        self._model_name = model_name
        
        # Get the start time (inherited from Emailer) 
        start_time = self.get_start_time()
        
        # Get the index of the Initialization Time since it is a categorical feature.
        # Used for the HistGradientBoostingClassifier. 
        features = list(X.columns)
        numeric_features = features if categorical_features is None else [i for i, f in enumerate(features) 
                                                                  if f not in categorical_features]
        
        # Get the base classifier.
        categorical_features_mask = [True for f in features if f in categorical_features]
        base_estimator = self._get_classifier(n_jobs=self._n_jobs, 
                                              categorical_features = categorical_features_mask) 
        
        # Build the pipeline. The pipeline is composed of 
        # an imputer (for any missing values), a min-max scaler, 
        # an random undersampler, and then a one-hot encoder 
        # for the categorical features (only init time at the moment).
        preprocessor = PreProcessPipeline(imputer='simple', 
                                      scaler='minmax',
                                      resample='under', 
                                      numeric_features=numeric_features, 
                                      categorical_features=categorical_features)
        
        steps = preprocessor.get_steps()
        steps.append(('model', base_estimator))
        pipeline = Pipeline(steps) 
       
        
        # Initialize CalibratedPipelineHyperOptCV. 
        known_skew = np.mean(y) 
         # Get the parameter grid for the hyperparameter tuning.
        param_grid = self.get_param_grid() if params is None else None

        print(f'{param_grid=}')
        
        clf = CalibratedHyperOptCV( estimator = pipeline,
                                  param_grid = param_grid,
                                  hyperopt='atpe',
                                  max_iter=20,
                                  scorer_kwargs = {'known_skew': known_skew},
                                  cv = 'date_based',
                                  cv_kwargs = {'n_splits' : 5,
                                               'dates' : dates,
                                               'valid_size' : 20 },
                                  )
            
        # Fit the model. 
        clf.fit(X,y, params)
            
        # Save the model. TODO: Might need to be more descript wiht the filename at some point. 
        save_fname = f'{model_name}_{time}_{target}.joblib'
        clf.save(join(self.ML_MODEL_SAVE_PATH, save_fname))
            
        # Send email 
        message = f'{join(self.ML_MODEL_SAVE_PATH, save_fname)} has finished training!'
        self.send_message(message, start_time) 
            
    
    def get_param_grid(self):
        if 'Random' in self._model_name:
             #RandomForest Grid
            param_grid = {
               'n_estimators': hp.choice('n_estimators', [100,250,300,500, 750, 1000]),
               'max_depth': hp.choice('max_depth', [3, 5, 8, 10, 15, 20,]),
               'max_features' : hp.choice( 'max_features', [5,6,8,10, 'sqrt']),
               'min_samples_split' : hp.choice( 'min_samples_split', [4,5,8,10,15,20,25,50] ),
               'min_samples_leaf': hp.choice( 'min_samples_leaf', [4,5,8,10,15,20,25,50]),

               }
        elif 'XGB' in self._model_name:
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

        elif 'Hist' in self._model_name:
            param_grid = {
                    'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
                    'max_iter' : hp.choice('max_iter', np.arange(100, 500, 50, dtype=int)),
                    'max_leaf_nodes' : hp.choice('max_leaf_nodes', [None, 1, 5, 10, 15, 20, 25, 30, 40]),
                    'min_samples_leaf': hp.choice( 'min_samples_leaf', [4,5,8,10,15,20,25,50]),                            
                    'l2_regularization': hp.choice( 'l2_regularization', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0]),
                    'early_stopping' : hp.choice('early_stopping', [True, False]), 
                 }
                                                

        elif 'Logistic' in self._model_name:
            param_grid = {
                'l1_ratio': hp.choice('l1_ratio', [0.0001, 0.001, 0.01, 0.1, 0.5, 0.6, 0.8, 1.0]),
                'C': hp.choice('C', [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]),
                }

        elif 'Network' in self._model_name:
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
    
                                                 
    def _get_classifier(self, n_jobs, **params):
        """Returns classifier machine learning model object."""
        # Random Forest Classifier 
        if  'RandomForestClassifier' in self._model_name:
            return RandomForestClassifier( n_jobs = n_jobs, criterion='entropy', random_state=42 )
        
        # Gradient-Boosted Tree Classifier 
        elif 'GradientBoostingClassifier' in self._model_name:
            return GradientBoostingClassifier()
        
        # Logistic Regression 
        elif 'LogisticRegression' in self._model_name:
            return LogisticRegression(n_jobs=n_jobs, solver='saga', penalty='elasticnet', max_iter=300, random_state=42)
        
        elif 'HistGradientBoosting' in self._model_name:
            try:
                categorical_features = params['categorical_features']
            except:
                raise KeyError('Expecting a categorical_features in params!') 
            return HistGradientBoostingClassifier(loss='binary_crossentropy', 
                                                  random_state=123, 
                                                  categorical_features=categorical_features)

        elif 'NeuralNetwork' in self._model_name:
            return MyNeuralNetworkClassifier()
        
        else:
            raise ValueError(f"{model_name} is not an accepted option!")
                                              
                                                 
        
# For fitting the baseline models (Calibrator)
class BaselineClassifier(Emailer):
    def __init__(self):
        pass 
                                                 
    
    
    
        