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

import sys
sys.path.insert(0, '/home/lucas.jones/frdd-wofs-ml-severe')
sys.path.insert(0, '/home/lucas.jones/python_packages/frdd-wofs-post')

# local imports
from wofs_ml_severe.fit.ml_trainer import MLTrainer

# necessary to prevent multiprocessing from infinitely looping as it spawns
if __name__ == "__main__":     

    print("============= STARTING A NEW TRAINING PIPELINE ============='")

    targets = ['severe_hail', 'severe_wind', 'severe_torn', 'severe_mesh']   #'sig_severe_hail',
            #'sig_severe_wind', 'any_severe', 'any_sig_severe', 'severe_warn', 'torn_warn', 
            #'hail', 'wind', 'mesh', 'tornado_probsevere']

    lead_times = ['first_hour', 'second_hour']    #'third_hour', 'fourth_hour']

    # train the models
    for target in targets:

        hazard = target.split('_')[0]

        if hazard == "hailsize":
            models = ["XGBRegressor", "XGBClassifier", "RFClassifier",
            "LogisticRegression"]
        else:
            models = ["XGBClassifier", "RFClassifier", "LogisticRegression"]  #"ElasticNet", "NNRegressor", "ExplainableBoostingRegressor"
            
        for model in models:
            for time in lead_times:

                # default to performing calibration (if a regression model MLTrainer automatically 
                # changes to false), hyperparameter tuning, no ensemble calibration (not all the 
                # models have been trained yet)
                trainer = MLTrainer(calibrate = True, hyopt_tune = True, ensemble_calibration = False,
                                    overwrite = False, debug = False)
                trainer.train_model(model, target, time)

    print("===================== Test Complete ======================")
