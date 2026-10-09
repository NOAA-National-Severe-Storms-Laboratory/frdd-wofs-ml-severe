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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(0, '/home/lucas.jones/frdd-wofs-ml-severe')
sys.path.insert(0, '/home/lucas.jones/python_packages/frdd-wofs-post')

# local imports
from wofs_ml_severe.fit.ml_trainer import MLTrainer

# necessary to prevent multiprocessing from infinitely looping as it spawns
if __name__ == "__main__":     

    print("============= STARTING A NEW TRAINING PIPELINE ============='")

    targets = ['severe_wind', 'severe_torn', 'severe_hail', 'severe_mesh']   #'sig_severe_hail', 
            #'sig_severe_wind', 'any_severe', 'any_sig_severe', 'severe_warn', 'torn_warn', 
            #'hail', 'wind', 'mesh', 'tornado_probsevere']

    lead_times = ['first_hour', 'second_hour']    #'third_hour', 'fourth_hour']
    
    # specify individual models. If None, the default models based on hazard type are used below
    models = ["NNRegressor"]      #["BaselineLR"]      
    VERSION = "fullfeat"          # the file_log or versioning information to add at the end of the model filename
    OVERWRITE = False              # whether or not to overwrite the existing models of the same configuration

    # train the models
    for target in targets:

        hazard = target.split('_')[0]

        if hazard == "hailsize" and models is None:
            models = ["XGBRegressor", "XGBClassifier", "RFClassifier", 
                      "LogisticRegression"]
        elif models is None:
            models = ["XGBClassifier", "RFClassifier", "LogisticRegression"]  #"ElasticNet", "NNRegressor", "ExplainableBoostingRegressor"
            
        for model in models:
            for time in lead_times:

                # default to performing calibration (if a regression model MLTrainer automatically 
                # changes to false), hyperparameter tuning, no ensemble calibration (not all the 
                # models have been trained yet)
                trainer = MLTrainer(calibrate = True, hyopt_tune = True, ensemble_calibration = False,
                                    overwrite = OVERWRITE, debug = False, file_log = VERSION)
                trainer.train_model(model, target, time)

    print("===================== Test Complete ======================")
