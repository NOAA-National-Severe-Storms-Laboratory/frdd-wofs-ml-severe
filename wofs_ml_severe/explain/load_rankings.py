import sys
sys.path.append('/home/monte.flora/python_packages/scikit-explain/')
sys.path.append('/home/monte.flora/python_packages/master/ml_workflow')
sys.path.append('/work/mflora/ROAD_SURFACE')
import skexplain
from os.path import join
import pickle
from glob import glob
import joblib

from probsr_config import PREDICTOR_COLUMNS, FIGURE_MAPPINGS, COLOR_DICT
from skexplain.common.importance_utils import to_skexplain_importance


# backward singlepass, forward multipass, coefs/gini, SHAP 
base_path = '/work/mflora/ML_DATA/'

def load_imp(hazard):

    # Load the WoFS-ML-Severe Models
    if hazard != 'road_surface':
        base_path = '/work/mflora/ML_DATA/MODEL_SAVES'
        model_name = 'LogisticRegression'
        model_paths = glob(join(base_path, f'{model_name}_first_hour_{hazard}*'))
        model_path = [m for m in model_paths if 'manual' not in m][0]
        model_data = joblib.load(model_path)

        model = model_data['model']
        feature_names = model_data['features']
    
    explainer = skexplain.ExplainToolkit()
    
    
    if hazard == 'road_surface':
        name = 'Random Forest'
    else:
        name = 'LogisticRegression'
    
    base_path = '/work/mflora/ML_DATA/'
    
    # permutation results
    if hazard == 'road_surface': 
        basePath = '/work/mflora/ROAD_SURFACE'
        bsp_fname = join(basePath,'permutation_importance', f'perm_imp_original_backward.nc')
        fmp_fname = join(basePath,'permutation_importance', f'perm_imp_original_forward.nc')
    else:    
        path = join(base_path, 'permutation_importance')
        bsp_fname = join(path, f'permutation_importance_{hazard}_first_hour_training_norm_aupdcbackward.nc' )
        fmp_fname = join(path, f'permutation_importance_{hazard}_first_hour_training_norm_aupdcforward.nc' )   

    bsp = explainer.load(bsp_fname)
    bmp = bsp.copy()
    fmp = explainer.load(fmp_fname)
    fsp = fmp.copy()
    

    # Backward singlepass and forward multipass: original_score - permuted score
    original_score = bsp[f'original_score__{name}'].values
    scores = original_score - bsp[f'singlepass_scores__{name}'].values
    bsp[f'singlepass_scores__{name}'] = (['n_vars_singlepass', 'n_permute'], scores)
    
    original_score = fmp[f'original_score__{name}'].values
    scores = original_score - fmp[f'multipass_scores__{name}'].values
    fmp[f'multipass_scores__{name}'] = (['n_vars_multipass', 'n_permute'], scores)

    # ALE     
    if hazard == 'road_surface':
        ale_var_fname = join(basePath,'ale_results', 'ale_var_rf_original.nc')
        ale_var = explainer.load(ale_var_fname)
        
    else:
        # ale results
        ale_fname = join(base_path,'ALE_RESULTS', f'ale_var_results_all_models_{hazard}_first_hour.nc')
        ale_var = explainer.load(ale_fname)
    
    
    
    if hazard == 'road_surface':
        # load the random forest
        rf = joblib.load(join(basePath, 'JTTI_ProbSR_RandomForest.pkl'))
        gini_values = rf.feature_importances_
        gini_rank = to_skexplain_importance(gini_values,
                                       estimator_name='Random Forest', 
                                       feature_names=PREDICTOR_COLUMNS, 
                                         method = 'gini')
        
    else:
        coefs = model.base_estimator.named_steps['model'].coef_[0]
        coef_rank = to_skexplain_importance(coefs,
                                       estimator_name=name, 
                                       feature_names=feature_names, 
                                        method = 'coefs')

    # shap results
    if hazard == 'road_surface':
        fname = join(basePath,'shap_results', 'shap_rf_original.nc')
        feature_names = PREDICTOR_COLUMNS
    else:
        fname = join(base_path, 'SHAP_VALUES', f'shap_values_LogisticRegression_{hazard}_first_hour.pkl')
    
    with open(fname, 'rb') as f:
        shap_data = pickle.load(f)
        shap_vals = shap_data['shap_values']
    
    shap_rank = to_skexplain_importance(shap_vals, 
                                      estimator_name=name, 
                                      feature_names=feature_names, 
                                      method ='shap_sum', )
    if hazard == 'road_surface':
        return ([bsp, bmp, fsp, fmp, gini_rank, shap_rank, ale_var],
                ['singlepass', 'multipass', 'singlepass', 'multipass', 'gini', 'shap_sum', 'ale_variance'], name)
                
    else:
        return ([bsp, bmp, fsp, fmp, coef_rank, shap_rank, ale_var],
                ['singlepass', 'multipass', 'singlepass', 'multipass', 'coefs', 'shap_sum', 'ale_variance'], name)