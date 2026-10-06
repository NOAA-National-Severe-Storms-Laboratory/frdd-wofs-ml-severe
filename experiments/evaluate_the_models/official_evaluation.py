#======================================================
# A copy (with some changes) of official_evaluation.ipynb 
# that is used to evaluate the ML models and produce 
# verification diagrams.
# 
# Author: Lucas Jones (Git username : LucasJ-NSSL)
# Email: lucas.jones@noaa.gov
# Date: Oct. 2, 2026
#======================================================

# The custom classifier 
import sys, os
sys.path.insert(0, '/home/lucas.jones/python_packages/WoF_post')
sys.path.insert(0, '/home/lucas.jones/python_packages/ml_workflow')
sys.path.insert(0, '/home/lucas.jones/frdd-wofs-ml-severe')
sys.path.insert(0, '/home/lucas.jones/python_packages/scikit-verify')

from wofs_ml_severe.io.io import MLDataLoader
from wofs_ml_severe.io.load_ml_models import load_ml_model
from wofs_ml_severe.common.emailer import Emailer 
from wofs.post.utils import load_yaml
from skverify.verification import plot_verification
from wofs_ml_severe.io.io import get_numeric_init_time

import numpy as np
from os.path import join
import matplotlib.pyplot as plt

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

OUTPATH = '/work2/lucas.jones/evaluation_mlsevere/'
DATA_PATH = '/work2/lucas.jones/ml_data/'
outname = "AllModels_evaluation.png"
fig_title = ""

names = ["XGBClassifier", "RFClassifier", "LogisticRegression"]     #'StackedEnsemble', 
resample = 'None'
lead_time = 'first_hour'
version = None
#target = 'wind_severe_0km'#['wind_severe_0km', 'hail_severe_0km', 'tornado_severe_0km']
#target_str = 'wind_severe_0km'#'all_severe'
if version is not None or not "":
    outname = outname.replace(".png", f"_{version}.png")

# a list of available targets and their corresponding thresholds and data 
# can be found in wofs_ml_severe.io.io.py
target = 'severe_hail'   #'severe_wind', 'severe_torn', 'severe_mesh', 'sig_severe_hail', 'sig_severe_wind',

eval_target = target      #same thing as target currently, was 'hail_severe_original'
target_str = get_target_str(target)
retro = False
# Evaluating the base classifier (without calibration)
append_base_est = False

BL_DICT = {'hail_severe_0km': 'hail_nmep_>1.0_0km__prob_max',
           'wind_severe_0km': 'wind_nmep_>40_0km__prob_max',
           'tornado_severe_0km' : 'uh_nmep_>180_0km__prob_max',
          }

# Load the ML models. 
ml_config = load_yaml(
    '/home/lucas.jones/frdd-wofs-ml-severe/wofs_ml_severe/conf/default_ml_config.yml')
models = []            # list of model objects
for name in names: 
    parameters = {
                'target' : target_str,
                'time' : lead_time, 
                'model_name' : name,
                'ml_config' : ml_config,
                'file_log' : version
            }

    model_in = load_ml_model(retro, **parameters)

    # extract the model and append it to the models list for later use
    # making predictions. Stacking Classifier models don't have a dictionary,
    # so it can be appended directly
    if name == "StackedEnsemble":
        models.append((name, model_in))
        features = model_in.features
    else:
        model = model_in['model']
        features = list(model_in["X"].columns)

        if append_base_est:
            models.append((name, model.calibrated_classifiers_[0].estimator["model"]))
        else:
            models.append((name, model))

# Load the data. 
data_loader = MLDataLoader(target_column=eval_target, 
                              lead_time=lead_time,
                              mode = "testing",
                              data_path = DATA_PATH)
X, y, metadata = data_loader.load()

'''
# loads baseline data, seemingly deprecated
X_bl, y, metadata = load_ml_data(target_col=eval_target, 
                                  lead_time=lead_time,
                                  mode=mode,
                                  baseline=True,
                                 base_path = DATA_PATH
                                 )
'''

X_test = X[features]
X_test = fix_data(X_test)
#X_test = get_numeric_init_time(X_test)

y_pred = [model.predict_proba(X_test)[:,1] for name, model in models]#[:-1]]

# prevent impossible negative predictions for regression models
for name in names:
    if "Regression" in name:
        print(name)      # test
        y_pred[y_pred[name] < 0.0] = 0.0

        style = "regression"         # useful for plot_verification later

    else:
        style = "classification"
 
#bl_pred = [models[-1].predict(X_bl.reshape(-1,1))]

#y_pred += bl_pred

#names = ['RF', 'LR', 'XB']     #, 'BL']     #['LR', 'BL'] 
fig, axes = plot_verification(models, X_test, y, n_boot = 10, style = style)
fig.suptitle(fig_title)
plt.savefig(join(OUTPATH, outname), dpi = 400)


