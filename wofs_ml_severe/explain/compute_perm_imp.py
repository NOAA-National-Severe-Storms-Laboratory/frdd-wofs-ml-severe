import sys
sys.path.append('/home/monte.flora/python_packages/scikit-explain')
sys.path.append('/home/monte.flora/python_packages/master/ml_workflow')

from load_models_and_data import load_model_and_data
import itertools 
from os.path import join, exists
from skexplain import ExplainToolkit
import numpy as np
from wofs_ml_severe.wofs_ml_severe.common.util import Emailer

""" usage: stdbuf -oL python -u compute_perm_imp.py 2 > & log_pimp & """

emailer = Emailer()
start_time = emailer.get_start_time()

n_permute = 30
subsample = 1.0
n_jobs=20 

random_state = np.random.RandomState(42)
N = 50000

path = '/work/mflora/ML_DATA/permutation_importance/'

lead_time = 'first_hour'
opts = ['', 'L1_based']
hazards = ['tornado', 'severe_hail', 'severe_wind']
directions = ['forward', 'backward']

for opt, hazard, direction in itertools.product(opts, hazards, directions):
    print(f'Hazard: {hazard}...Opt: {opt}...Direction: {direction}')
    
    results_fname = join(path, f'perm_imp_{hazard}_{lead_time}_{opt}_{direction}.nc')
    if exists(results_fname):
        continue
    
    estimator, X,y = load_model_and_data(hazard, opt)
    explainer = ExplainToolkit(estimator,X,y)
    
    inds = random_state.choice(len(X), N, replace=False) 

    X = X.iloc[inds,:]
    X.reset_index(drop=True, inplace=True)
    y = y[inds]
    
    results = explainer.permutation_importance(n_vars=X.shape[1], 
        evaluation_fn='norm_aupdc', 
        n_permute=n_permute,
        subsample=subsample,
        n_jobs=n_jobs,
        direction=direction,
        verbose=True, 
        return_iterations=False, 
        )

    print(f'Saving {results_fname}...')
    explainer.save(fname=results_fname, data=results)
    
    emailer.send_message(f" Perm. Imp. for {hazard}, {opt}, {direction} is done!", start_time)
