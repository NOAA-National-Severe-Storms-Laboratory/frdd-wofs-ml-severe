import joblib, os, warnings
from os.path import join 

def load_ml_model(retro=False, **parameters):
    """
    Load a saved ML model  
    """
    ml_config = parameters.get('ml_config') 

    PATH = ml_config['ML_MODEL_PATH']
   
    time = parameters.get('time', 'first_hour')
    target = parameters['target']
    file_log = parameters.get('file_log', None)
    model_name = parameters['model_name']

    scaler = 'standard' if model_name in ["LogisticRegression", 'NeuralNetwork'] else None
    
    #if retro:
        #resample = ml_config['RESAMPLE_DICT'][time][target][model_name]
    #    resample=None
    #    model_fname = f'{model_name}_{time}_{target}_{resample}_{scaler}_{drop_opt}.pkl'
    #else:
    
    retro_str = 'retro' if retro else 'realtime'
    
    resample = None 
    model_fname = f'{model_name}_{target}_{resample}_{time}_{retro_str}.joblib'
    
    if file_log is not None:
        model_fname = model_fname.replace('.joblib', f'__{file_log}.joblib')
    
    print(f'Loading {join(PATH, model_fname)}...')
    
    model = joblib.load(join(PATH, model_fname))
    

    return model

def load_calibration_model(retro=False, **parameters):
    """
    Load the isotonic model, which is used to calibrate 
    the baseline product developed for comparison against 
    the WoFS-ML-Severe products.
    """
    ml_config = parameters.get('ml_config') 
    PATH = ml_config['ML_MODEL_PATH'] 
    baseline_vars = ml_config['BASELINE_VARS']
    
    time = parameters.get('time', 'first_hour')
    target = parameters['target']
    if retro:
        save_fname = f'calibration_model_wofs_{time}_{target}_{baseline_vars[target]}.joblib'
    else:
        save_fname =  f'Baseline_{target}_None_{time}_realtime.joblib'
        
    data = joblib.load(join(PATH, save_fname))
    model = data['model']
    
    return model
    


