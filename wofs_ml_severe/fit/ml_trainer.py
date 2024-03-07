from os.path import join, exists
import os
import itertools
import multiprocessing as mp 
import numpy as np

# The custom classifier 
import sys
sys.path.insert(0, '/home/monte.flora/python_packages/wofs_ml_severe')
sys.path.insert(0, '/home/monte.flora/python_packages/ml_workflow')
sys.path.insert(0, '/home/monte.flora/python_packages/WoF_post')

from ..common.emailer import Emailer 
from ..io.io import MLDataLoader

from ml_workflow import TunedEstimator 
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from datetime import datetime
import traceback
from numba import cuda
import gc

from .ml_configuration import MLConfiguration

class MLTrainer:
    
    BASELINE_VARS = ['uh_2to5_instant__time_max__amp_ens_mean_spatial_perc_90',
           'ws_80__time_max__amp_ens_mean_spatial_perc_90',
           'hailcast__time_max__amp_ens_mean_spatial_perc_90',
          ]
    
    BASELINE_NMEP_VARS = ['hail_nmep_>1.0_0km__prob_max',
                     'wind_nmep_>40_0km__prob_max',
                     'uh_nmep_>180_0km__prob_max',
                            ]
    
    OBJECT_FEATURES = ['area','eccentricity', 'extent', 'orientation',
                       'minor_axis_length', 'major_axis_length', 'ens_track_prob',
                       'area_ratio']

    # Since logistic regression has a small search space, we can use simple GridSearchCV
    # for the hyperparameter search. 
    SMALL_SEARCH_SPACES = [ "LogisticRegression", "BaselineLR", "LinearRegression", 'ElasticNet']
    
    REGRESSION_MODELS = ['ElasticNet', 'XGBRegressor', 'RFRegressor', 'NNRegressor', 
                        'ExplainableBoostingRegressor']
    
    CLASS_MODELS = ['LogisticRegression', 'BaselineClass',  'XGBClassifier', 'NNClassifier', 
                   'ExplainableBoostingClassifier']
    
    IS_KERAS_MODEL = ['NNRegressor', 'NNClassifier']
    
    
    def __init__(self, outpath='/work/mflora/ML_DATA/NEW_ML_MODELS', 
                 calibrate=False, hyopt_tune=False, ensemble_calibration=False, 
                 scaler='standard',  overwrite=False, 
                 debug=False, file_log=None, scorer=None, hyperopt_optimizer=None,
                loader_kws= {
                             'data_path' :  '/work/mflora/ML_DATA/DATA/',
                             'return_full_dataframe': False, 
                             'random_state' : 123, 
                             'months' : ['April', 'May', 'June'],
                             'years' : [2018, 2019, 2020, 2021, 2022],  
                             'mode' : 'training'
                            }):
        
        self.outpath = outpath
        self.calibrate = calibrate
        self.hyopt_tune = hyopt_tune 
        self.ensemble_calibration = ensemble_calibration 
        self.scaler = scaler 
        self.loader_kws = loader_kws
        self.debug = debug
        self.file_log = file_log 
        self.hyperopt_optimizer = hyperopt_optimizer
        
        self.overwrite = overwrite
        self.scorer = scorer
        self.emailer = Emailer()
        
        
    def train_model(self, model_name, target, lead_time, sample_weight=False):
        """Train the ML pipeline"""
        # The multiprocessing is neccesary to free up the GPU space
        # when training the XGBoost and NeuralNetwork models.
        self.sample_weight=sample_weight
        def fitting():
            self.is_keras = self.is_keras_model(model_name)
            self.target_type = self.get_target_type(model_name)
            
            if self.target_type == 'regression' and self.calibrate:
                self.calibrate = False 
                print('Setting calibrate=False as it can only be applied to Classification Models')

            start_time = self.emailer.get_start_time()
            # Build a file path for the ML model and the hyperopt results. 
            ml_fname = self.get_filename(self.outpath, model_name, target, lead_time, ext='joblib')
            subject = self.get_email_subject(model_name, target, lead_time, ml_fname)
            hyp_path = join(self.outpath, 'hyperopt_results')
            hyp_fname = self.get_filename(hyp_path, model_name, target, lead_time, ext='json')
            
            
            if not self.overwrite:
                if exists(ml_fname):
                    return None
                
            # Get the configuration params for the ML models.
            config = MLConfiguration.get_config(model_name)
            
            # Load the ML Data.
            X, y, metadata = self.load_data(model_name, target, lead_time)

            # Get the CV params 
            cv, groups = self.get_cv_params(X,y,metadata)
        
            # Get HPO, pipeline, and calibration params
            hyperopt_kwargs = self.get_hyperopt_kws(model_name, cv, config, hyp_fname)
            pipeline_kwargs = self.get_pipeline_kws()
            calibration_cv_kwargs = self.get_calibration_kws(cv, config)
        
            if self.sample_weight:
                sample_weight = self.get_sample_weight(y)
            else:
                sample_weight = None
        
            result = self.fit_and_save(X, y, groups, sample_weight, config, hyperopt_kwargs, 
                              pipeline_kwargs, calibration_cv_kwargs, ml_fname)
        
            if result:
                try:
                    self.emailer.send_email(subject, start_time)
                except:
                    print('Unable to send email. Possibly an NSSL network issue.')
        
        fitting_process = mp.Process(target=fitting)
        fitting_process.start()
        fitting_process.join()
    
    def get_sample_weight(self, y, threshold=0):
        return np.where(y > threshold, 5, 1)
    
    def is_keras_model(self, model_name):
        return model_name in self.IS_KERAS_MODEL
    
    def get_target_type(self, model_name):
        
        if model_name  in self.CLASS_MODELS:
            return 'classification'
        
        elif model_name in self.REGRESSION_MODELS:
            return 'regression'
        
        else:
            raise ValueError(f'{model_name} is not in CLASS_MODELS or REGRESSION_MODELS!')
    
    def get_email_subject(self, model_name, target, lead_time, fname):
        """Get the message for the email"""
        print('\nTraining a new model....')
        subject = f"""Target: {target} 
          Model Name : {model_name} 
          Lead Time: {lead_time}
          File : {fname}
          \n"""
        print(subject)
        return subject
    
    def get_filename(self, path, model_name, target, lead_times, ext='joblib'):
        """Create the save name for the ML model."""
        # ISO 8601 timestamp format, safe for filenames
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Replace spaces or other unsafe characters in model name and target variable
        model_name_safe = model_name.replace(' ', '_')

        if isinstance(target, list):
            target_safe = '_'.join(str(t) for t in target)
        else:
            target_safe = target.replace(' ', '_')
        
        # Convert lead times list to a string, joined by underscores
        if isinstance(lead_times, list):
            lead_times_str = '_'.join(str(lead_time) for lead_time in lead_times)
        else:
            lead_times_str = str(lead_times)
        
        # Construct the filename
        rs = self.loader_kws['random_state']
        
        filename = f"{model_name_safe}_{target_safe}_{lead_times_str}_rs_{rs}.{ext}"
        
        if self.file_log is not None:
            filename = filename.replace(f'.{ext}', f'_{self.file_log}.{ext}')
        
        return join(path, filename)
    
    
    def fit_and_save(self, X, y, groups, sample_weight, config, 
                     hyperopt_kwargs, pipeline_kwargs, calibration_cv_kwargs, fname):
        # Fit the model and save it.
        try: 
            model = config['model'](**config.get('model_params', {}))
            estimator = TunedEstimator(model, 
                                       pipeline_kwargs, hyperopt_kwargs, calibration_cv_kwargs)
        except Exception as e:
            print(f'Model Training Error {traceback.format_exc()}')
            return False
    
        if hasattr(y, 'values'):
            y = y.values
    
        estimator.fit(X,y,groups,sample_weight)
        estimator.save(fname, keras=self.is_keras)
        del estimator, X, y
        
        gc.collect()

        try:
            cuda.select_device(0)
            cuda.close()
        except:
            print('TunedEstimator Warning: failed to find GPU and deallocate memory.')
        
        return True
    
    def load_data(self, model_name, target, time):
        """Load the input ML dataframe, target variable, and metadata"""
        self.loader_kws['target_column'] = target
        self.loader_kws['lead_time'] = time 
        loader = MLDataLoader(**self.loader_kws) 
        
        X, y, metadata = loader.load()
        
    
        if model_name == 'BaselineLR':
            # Reduced to 3 variables (hail, wind, UH). 
            X = self.get_baseline_features(X)
            bl_loader_kws = self.loader_kws.copy()
            
            bl_loader_kws['load_baseline_dataframe'] = True
            bl_loader = MLDataLoader(**bl_loader_kws) 
            
            X_bl = bl_loader.load() 
            
            # Add on the baseline variables. 
            X[self.BASELINE_NMEPVARS] = X_bl[self.BASELINE_NMEP_VARS]
        
    
    
        if self.debug:
            inds = np.random.choice(len(X), size=20000)
        
            X = X.iloc[inds]
            y = y[inds]
            metadata = metadata.iloc[inds]
    
            X.reset_index(inplace=True, drop=True)
            metadata.reset_index(inplace=True, drop=True)
    
    
        return X, y, metadata

    def get_calibration_kws(self, cv, config):
        """Initialize the kwargs for the calibration model.""" 
        calibration_cv_kwargs = None
        if self.calibrate:
            calibration_cv_kwargs = {'method' : 'isotonic', 
                                     'ensemble' : self.ensemble_calibration, 
                                     'cv' : cv, 
                                     'n_jobs': config['n_jobs']}
        return calibration_cv_kwargs
        
    def get_pipeline_kws(self):
        """Get the pipeline kwargs"""
        # Initialize the kwargs for the Pipeline. 
        pipeline_kwargs={'imputer' : 'simple', 
                     'resample': None, 
                     'scaler': self.scaler,  
                        }
        #             'numeric_features' : numeric_features, 
        #             'categorical_features' :  categorical_features}
        
        return pipeline_kwargs
        
    
    def get_hyperopt_kws(self, model_name, cv, config, output_fname):
        """Initialize the kwargs for the hyperparameter optimization"""
        if self.hyperopt_optimizer is None:
            optimizer = 'grid_search' if model_name in self.SMALL_SEARCH_SPACES else 'tpe'
        else:
            optimizer = self.hyperopt_optimizer
        
        if model_name in self.SMALL_SEARCH_SPACES or optimizer in ['grid_search', 'random_search']:
            if self.scorer is None:
                scorer = 'average_precision' if self.target_type=='classification' else 'neg_root_mean_squared_error'
            else:
                scorer = self.scorer
        else:
            if self.scorer is None:
                scorer = MLConfiguration.get_scorer(self.target_type)
            else:
                scorer = self.scorer

        hyperopt_kwargs = None
        if self.hyopt_tune:       
            hyperopt_kwargs = {'search_space' : config['search_space'], 
                   'optimizer' : optimizer, 
                   'max_evals' : config['n_iter'], 
                   'patience' : config['patience'], 
                  'scorer' : scorer, 
                  'n_jobs' : config['n_jobs'], 
                  'cv' : cv, 
                  'output_fname' : output_fname 
                      }
            
        return hyperopt_kwargs
        
    def get_cv_params(self, X,y, metadata):
        """Get the Cross-validation parameters"""
        dates = metadata['Run Date']
        
        # Determine the example grouping for the cross-validation. 
        groups = self.dates_to_groups(dates, n_splits=5)
        
        # Initialize the cross-validation groups 
        cv = list(StratifiedGroupKFold(n_splits=5).split(X,y,groups))
        
        return cv, groups
        
    def dates_to_groups(self, dates, n_splits=5): 
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
    
  

    def get_baseline_features(self, X):
        """For the baseline method, using a logistic regression model
          trained on mid-level UH, 80-m wind speed, and hailcast.""" 
        return X[self.BASELINE_VARS]
