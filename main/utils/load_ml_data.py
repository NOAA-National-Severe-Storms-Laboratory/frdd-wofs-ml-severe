import pandas as pd


base_path = '/work/mflora/ML_DATA/MLDATA'

def load_ml_data(hazard, lead_time, mode):
    df = pd.read_feather(join(base_path, f'wofs_ml_severe__{time}__{mode}_baseline_data.feather'))
    return df