import sys
sys.path.append('/home/monte.flora/python_packages/master/ml_workflow')
sys.path.append('/work/mflora/ROAD_SURFACE')
from os.path import join
from glob import glob
import joblib
import pandas as pd

from probsr_config import PREDICTOR_COLUMNS,  TARGET_COLUMN
from calibration_classifier import CalibratedClassifier

def load_model_and_data(hazard, opt='', test=False):
    if hazard != 'road_surface':
        name = 'LogisticRegression'
        base_path = '/work/mflora/ML_DATA/MODEL_SAVES'
        model_name = 'LogisticRegression'
        model_paths = glob(join(base_path, f'{model_name}_first_hour_{hazard}*'))
        if opt != '':
            model_path = [m for m in model_paths if 'manual' in m][0]
        else:
            model_path = [m for m in model_paths if 'manual' not in m][0]
        print(f'Loading {model_path}...')
        model_data = joblib.load(model_path)

        model = model_data['model']
        feature_names = model_data['features']

        print(f'Loading data...') 
        base_path = '/work/mflora/ML_DATA/DATA'
        data_path = join(base_path, f'original_first_hour_training_matched_to_{hazard}_0km_data.feather')
        df = pd.read_feather(data_path)
        
        X = df[feature_names].astype(float)
        y = df[f'matched_to_{hazard}_0km'].astype(float)
        
    else:
        name = 'Random Forest'
        base_path = '/work/mflora/ROAD_SURFACE'
        model = joblib.load(join(base_path, 'JTTI_ProbSR_RandomForest.pkl'))
        #calibrator = joblib.load(join(base_path, 'JTTI_ProbSR_RandomForest_Isotonic.pkl'))
        #model = CalibratedClassifier(model, calibrator)
        
        if test:
            df = pd.read_csv(join(base_path, 'probsr_testing_data.csv'))
        else:
            df = pd.read_csv(join(base_path, 'probsr_training_data.csv'))
        
        X = df[PREDICTOR_COLUMNS].astype(float)
        y = df[TARGET_COLUMN].astype(float) 

    return (name,model), X, y

