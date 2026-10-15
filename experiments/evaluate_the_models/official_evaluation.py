#======================================================
# A copy (with significant changes) of official_evaluation.ipynb 
# that is used to evaluate the ML models and produce 
# verification diagrams.
# 
# Author: Lucas Jones (Git username : LucasJ-NSSL)
# Email: lucas.jones@noaa.gov
# Date: Oct. 2, 2026
#======================================================

import sys, os
 # -1 forces CPU usage otherwise GPU automatically used due to downstream tensorflow import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      

sys.path.insert(0, '/home/lucas.jones/python_packages/WoF_post')
sys.path.insert(0, '/home/lucas.jones/python_packages/ml_workflow')
sys.path.insert(0, '/home/lucas.jones/frdd-wofs-ml-severe')
sys.path.insert(0, '/home/lucas.jones/python_packages/scikit-verify')

from wofs_ml_severe.io.io import MLDataLoader, HailSizeLoader
from wofs_ml_severe.io.load_ml_models import load_ml_model
from wofs.post.utils import load_yaml
from skverify.verification import plot_verification
from wofs_ml_severe.io.io import get_numeric_init_time

import multiprocessing as mp
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

def get_target_str(target):
    # Initialize the kwargs for the hyperparameter optimization.
    if isinstance(target, list):
        if 'sig_severe' in target[0]:
            target = 'all_sig_severe'
        else:
            target = 'all_severe'
   
    return target 

def fix_data(X): 
    #X = X.astype({'Initialization Time' : str})
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.reset_index(inplace=True, drop=True)
    
    return X 

# , "BaselineLR",  'StackedEnsemble', "NNClassifier", "XGBClassifier", "RFClassifier", "LogisticRegression"
names = ["NNRegressor", "RFRegressor", "XGBRegressor"]     
lead_times = ['first_hour', 'second_hour']
MODIFIER = "AllModels"        # the model name or combination to list at the start of the output file
VERSION = "fullfeat"          # if desired model to retrieve has multiple versions, set name convention here
TASK = "regression"           # determines whether classification or regression evaluation is performed

# a list of available targets and their corresponding thresholds and data 
# can be found in wofs_ml_severe.io.io.py
# 'severe_mesh', 'sig_severe_hail', 'sig_severe_wind', 'severe_hail', 'severe_wind', 'severe_torn'
targets = ["hail_size"]

OUTPATH = '/work2/lucas.jones/evaluation_mlsevere/'
DATA_PATH = '/work2/lucas.jones/ml_data/'

# the number of bootstrapping iterations for confidence intervals of the 
# metrics(see 2021 paper, pg. 10 for caveat)
n_boot = 1000         

for target in targets:

    if target == "torn_warn":
        target = "severe_torn"
        eval_target = "torn_warn"
    else:
        eval_target = target
        
    for lead_time in lead_times:

        # make the outfile name based on the target and version
        if VERSION is not None:
            outname = f"{MODIFIER}_{target}_{lead_time}_eval_{VERSION}.png"
        else:
            outname = f"{MODIFIER}_{target}_{lead_time}_eval.png"

        target_str = get_target_str(target)
        retro = False
        append_base_est = False       # Evaluating the base classifier (without calibration)
        baseline = False              # flag for later baseline related calculations

        BL_DICT = {'hail_severe_0km': 'hail_nmep_>1.0_0km__prob_max',
                'wind_severe_0km': 'wind_nmep_>40_0km__prob_max',
                'tornado_severe_0km' : 'uh_nmep_>180_0km__prob_max',
                }

        # Load the ML models. 
        ml_config = load_yaml(
            '/home/lucas.jones/frdd-wofs-ml-severe/wofs_ml_severe/conf/default_ml_config.yml')

        models = []            # list of model objects and names in a tuple
        bl_models = []         # a list of baseline model objects and names in a tuple

        for name in names: 
            parameters = {
                        'target' : target_str,
                        'time' : lead_time, 
                        'model_name' : name,
                        'ml_config' : ml_config,
                        'file_log' : VERSION
                    }

            model_in = load_ml_model(retro, **parameters)

            # extract the model and append it to the models list for later use
            # making predictions. Stacking Classifier models don't have a dictionary,
            # so it can be appended directly. Any baseline model should be added 
            # to the bl_models list
            if "Baseline" in name:
                bl_models.append((name, model_in['model']))
                bl_features = list(model_in["X"].columns)

                baseline = True  
            elif name == "StackedEnsemble":
                models.append((name, model_in))
                features = model_in.features
            else:
                model = model_in['model']
                features = list(model_in["X"].columns)

                if append_base_est:
                    models.append((name, model.calibrated_classifiers_[0].estimator["model"]))
                else:
                    models.append((name, model))

        # Load the data, with specific condition to use HailDataLoader in the case of 
        # predicting hail_size
        if target == "hail_size":
            data_loader = HailSizeLoader(lead_time=lead_time, mode="testing")
        else:
            data_loader = MLDataLoader(target_column=eval_target, 
                                        lead_time=lead_time,
                                        mode = "testing",
                                        data_path = DATA_PATH)
            
        X, y, metadata = data_loader.load()

        X_test = X[features]
        X_test = fix_data(X_test)

        # if a baseline model is included, subset the features to those used by baselines otherwise 
        # set the bl_model list and data to None for integration with plot_verifiction
        if baseline:
            X_bl = X[bl_features]
        else: 
            bl_models = None
            X_bl = None

        # plot_verification does not in reality actually support regression plots (w/o making 
        # significant changes). As a result, if statements used to calculate different metrics 
        # and produce plots appropriate to either classification or regression
        if TASK =="classification":
            y_pred = {name : model.predict_proba(X_test)[:,1] for name, model in models}#[:-1]]
            
            fig, axes = plot_verification(models, X_test, y, n_boot = n_boot, baseline_estimators = bl_models,
                                        X_baseline = X_bl, style = "classification")
            # create a figure title based on the many possible configurations
            hour = lead_time.split("_")[0]
            if VERSION is not None:
                fig_title = f"Evaluation of {hour} hour {eval_target} with {VERSION}"    
            else:
                fig_title = f"Evaluation of {hour} hour {eval_target}"       

            fig.suptitle(fig_title)
            plt.savefig(join(OUTPATH, outname), dpi = 400)
            plt.close()

        elif TASK == "regression":

            metrics = []
            for name, model in models:
                y_pred = model.predict(X_test)#[:-1]]

                # clamping to prevent impossible negative predictions for regression models
                y_pred[y_pred < 0.0] = 0.0 

                # calculate metrics such as mse, mae
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                medae = median_absolute_error(y, y_pred)

                # output as a Pandas table
                metrics.append([name, mse, mae, medae])
                
            df = pd.DataFrame(metrics, columns = ["Model Name", "MSE", "MAE", "MedAE"])
            print(df)
        
    