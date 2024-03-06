# The custom classifier 
import sys, os
sys.path.insert(0, '/home/monte.flora/python_packages/ml_workflow')


import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import RootMeanSquaredError

from ml_workflow.custom_losses import (WeightedMSE, 
                                              RegressLogLoss_Normal, 
                                              RegressLogLoss_SinhArcsinh,
                                              CustomHailSizeLoss, 
                                       MyWeightedMSE)

# custom evaluation metrics. 
from ml_workflow.custom_metrics import (ParaRootMeanSquaredError2, 
                                        ConditionalRootMeanSquaredError, 
                                        ConditionalParaRootMeanSquaredError,
                                        CSIScoreThreshold
                                       )

from ml_workflow.tf_pipeline import TensorFlowPipeline

def load_tf_model(tf_model_path = '/work/mflora/ML_DATA/NN_MODELS/custom_loss_tune/model_56.h5'):
    
    base_path = os.path.dirname(tf_model_path)
    
    tf_model = load_model(tf_model_path, custom_objects = 
                              {
                               'RegressLogLoss_SinhArcsinh' : RegressLogLoss_SinhArcsinh,
                               'RegressLogLoss_Normal' : RegressLogLoss_Normal,
                               'WeightedMSE' : WeightedMSE,
                               'ParaRootMeanSquaredError2': ParaRootMeanSquaredError2, 
                               'ConditionalRootMeanSquaredError': ConditionalRootMeanSquaredError, 
                               'ConditionalParaRootMeanSquaredError': ConditionalParaRootMeanSquaredError,
                               'CustomHailSizeLoss' : CustomHailSizeLoss,
                               'MyWeightedMSE': MyWeightedMSE,
                               'CSIScoreThreshold' : CSIScoreThreshold  
                              })

    # This function assumes that the preprocesing sklearn models are available in the same 
    # directory as the tensorflow model. 
    names = ['imputer', 'scaler']
    preprocessors = [joblib.load(os.path.join(base_path, f'{name}.joblib')) 
                 for name in names]
    pipeline = TensorFlowPipeline(tf_model, preprocessors)
    
    return pipeline 