import joblib, os, warnings
from os.path import join 

from ..fit.ml_trainer import MLTrainer
from .load_tf_model import load_tf_model

def load_ml_models_2024(ml_model_path, return_features=False): 

    if ml_model_path.endswith('.joblib'): 
        model_dict = joblib.load(ml_model_path)
        model, features = model_dict['model'], model_dict['features'] 
        # fname format : 'WeightedAvgClassifer__any_severe.joblib' 
        model_fname = os.path.basename(ml_model_path)
        fname_no_ext = model_fname.replace('.joblib', '') 
        _, target = fname_no_ext.split('__')

        if return_features:
            return model, target, features 
        else:
            return model, target 
    
    else:
        # Loading the regression model. 
        model = load_tf_model(ml_model_path)
        
        return model, 'hail_size'
        
        
def load_ml_model(retro=False,  **parameters):
    """
    Load a saved ML model  
    """
    path = parameters.get('model_path', '/work/mflora/ML_DATA/OPERATIONAL_MODELS_2023')
    ml_config = parameters.get('ml_config', {})
    if path is None:
        path = '/work/mflora/ML_DATA/OPERATIONAL_MODELS_2023'
        
    if not os.path.exists(path):
        print(f"{path} does not exist! Reverting to /work/mflora/ML_DATA/OPERATIONAL_MODELS_2023 ")
        path = '/work/mflora/ML_DATA/OPERATIONAL_MODELS'

    time = parameters.get('time', 'first_hour')
    target = parameters['target']
    file_log = parameters.get('file_log', None)
    if file_log is None:
        file_log = ml_config.get('file_log', None)
    
    random_state = parameters.get('random_state', 123)
    
    model_name = parameters['model_name']
    
    old_file_format = parameters.get('old_file_format', False)
    

    if old_file_format: 
        scaler = 'standard' if model_name in ["LogisticRegression", 'NeuralNetwork'] else None
    
        #if retro:
            #resample = ml_config['RESAMPLE_DICT'][time][target][model_name]
        #    resample=None
        #    model_fname = f'{model_name}_{time}_{target}_{resample}_{scaler}_{drop_opt}.pkl'
        #else:
    
        retro_str = 'retro' if retro else 'realtime'
    
        resample = parameters.get('resample', None) 
    
        model_fname = f'{model_name}_{target}_{resample}_{time}_{retro_str}.joblib'
    
        if file_log is not None:
            model_fname = model_fname.replace('.joblib', f'__{file_log}.joblib')
    
    else:
        model_fname = f'{model_name}_{target}_{time}_rs_{random_state}.joblib'
    
    print(f'Loading {join(path, model_fname)}...')
    model = joblib.load(join(path, model_fname))
    
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
    


