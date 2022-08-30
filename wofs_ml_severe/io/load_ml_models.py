import joblib, os, warnings
from os.path import join 

def load_ml_model(**parameters):
    """
    Load a saved ML model  
    """
    ml_config = parameters.get('ml_config') 

    PATH = ml_config['ML_MODEL_PATH']
   
    time = parameters.get('time', 'first_hour')
    target = parameters['target']
    drop_opt = parameters['drop_opt']
    model_name = parameters['model_name']

    scaler = 'standard' if model_name in ["LogisticRegression", 'NeuralNetwork'] else None
    resample = ml_config['RESAMPLE_DICT'][time][target][model_name]

    model_fname = f'{model_name}_{time}_{target}_{resample}_{scaler}_{drop_opt}.pkl'
    
    model = joblib.load(join(PATH, model_fname))

    return model

def load_calibration_model(**parameters):
    """
    Load the isotonic model, which is used to calibrate 
    the baseline product developed for comparison against 
    the WoFS-ML-Severe products.
    """
    ml_config = parameters.get('ml_config') 
    PATH = ml_config['ML_MODEL_PATH'] 
    
    baseline_var = {'tornado' : 'uh_probs_>180_prob_max',
        'severe_hail' : 'hail_probs_>1.0_prob_max',
        'severe_wind' : 'wnd_probs_>40_prob_max'
    }
    time = parameters.get('time', 'first_hour')
    target = parameters['target']

    save_fname = f'calibration_model_wofs_{time}_{target}_{baseline_var[target]}.joblib'
    iso_reg = joblib.load(join(PATH, save_fname))
    
    return iso_reg
    


