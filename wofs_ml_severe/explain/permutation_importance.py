import numpy as np
import pandas as pd
import joblib
import sys, os
sys.path.insert(0, '/home/monte.flora/python_packages/WoF_post')
sys.path.insert(0, '/home/monte.flora/python_packages/ml_workflow')
sys.path.insert(0, '/home/monte.flora/python_packages/wofs_ml_severe')

from wofs_ml_severe.io.load_ml_models import load_ml_model
from wofs.post.utils import load_yaml
import skexplain 
from skexplain.common.importance_utils import to_skexplain_importance 
from display_names import to_display_name, to_units, to_color, map_to_readable_names

time = 'first_hour'
target = 'hail_severe_0km'
retro=True

ml_config = load_yaml(
    '/home/monte.flora/python_packages/wofs_ml_severe/wofs_ml_severe/conf/ml_config_retro.yml')

parameters = {
                'target' : target,
                'time' : time, 
                'drop_opt' : '',
                'model_name' : 'LogisticRegression',
                'ml_config' : ml_config,
            }

model_dict = load_ml_model(retro, **parameters)
model = model_dict['model']


ml_config = load_yaml(
    '/home/monte.flora/python_packages/wofs_ml_severe/wofs_ml_severe/conf/ml_config_retro.yml')

parameters['model_name'] = 'LogisticRegression'
parameters['ml_config'] = ml_config
model_dict = load_ml_model(retro, **parameters)
model = model_dict['model']
X =  model_dict['X']
y = model_dict['y']

rs = np.random.RandomState(123)





explainer = skexplain.ExplainToolkit(('ML', model), X=X, y=y,)



