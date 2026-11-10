#======================================================
# This script uses the newer ML training classes in ml_trainer.py
# and stacking_classifier.py from /wofs_ml_severe/fit/ 
# and /wofs_ml_severe/common respectively to train the 
# ML Severe models.
# 
# Author: Lucas Jones (Git username : LucasJ-NSSL)
# Email: lucas.jones@noaa.gov
# Date: Sept. 28, 2026
#======================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(0, '/home/lucas.jones/frdd-wofs-ml-severe')
sys.path.insert(0, '/home/lucas.jones/python_packages/frdd-wofs-post')

# local imports
from wofs_ml_severe.fit.ml_trainer import MLTrainer

# necessary to prevent multiprocessing from infinitely looping as it spawns
if __name__ == "__main__":     

    print("============= STARTING A NEW TRAINING PIPELINE ============='")

    targets = ['severe_mesh', 'severe_wind', 'severe_torn', 'any_severe', 'any_sig_severe']     #'sig_severe_hail',  'severe_hail'
            #'sig_severe_wind', , 'severe_warn', 'torn_warn', "hail_size"
            #'hail', 'wind', 'mesh', 'tornado_probsevere']

    lead_times = ['first_hour', 'second_hour', ]    #'third_hour', 'fourth_hour'
    
    # specify individual models. If None, the default models based on hazard type are used below
    models = ["NNClassifier"]      #"XGBClassifier", "RFClassifier", "LogisticRegression", "XGBRegressor", "BaselineLR", "NNRegressor", , "RFRegressor"
    VERSION = "eqhaz"          # the file_log or versioning information to add at the end of the model filename
    OVERWRITE = False              # whether or not to overwrite the existing models of the same configuration

    custom_kws = {
    'data_path' : '/work2/lucas.jones/ml_data/',
    'return_full_dataframe': False, 
    'random_state' : 123, 
    'months' : ['April', 'May', 'June'],
    'years' : [2026],  
    'mode' : 'training', 
    'version': VERSION}

    # train the models
    for target in targets:

        if target == "hail_size" or target == "mesh" and models is None:
            models = ["XGBRegressor", "RFRegressor", "NNRegressor"]
        elif models is None:
            models = ["NNClassifier", "XGBClassifier", "RFClassifier", "LogisticRegression"]  
            
        for model in models:
            for time in lead_times:
                # default to performing calibration (if a regression model MLTrainer automatically 
                # changes to false), hyperparameter tuning, no ensemble calibration (not all the 
                # models have been trained yet)
                trainer = MLTrainer(calibrate = False, hyopt_tune = True, ensemble_calibration = False,
                                    overwrite = OVERWRITE, debug = False, file_log = VERSION, 
                                    loader_kws = custom_kws)
                trainer.train_model(model, target, time)

    print("===================== Training Complete ======================")
