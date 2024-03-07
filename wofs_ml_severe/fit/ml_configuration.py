# Import ML models 
# Sklearn 
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import average_precision_score, mean_squared_error
from typing import Dict, Iterable, Any
import scipy.stats as stats

try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
except:
    print('interpret-ml not installed')
    ExplainableBoostingRegressor = None

try:
    from scikeras.wrappers import KerasClassifier, KerasRegressor
except:
    print('scikeras is not installed')
    
try:
    import tensorflow as tf

    from tensorflow import keras
    from tensorflow.keras.models import Model, save_model, load_model
    from tensorflow.keras.layers import (Dense, 
                                     Activation, 
                                     Conv2D, 
                                     Conv3D,  
                                     Input, 
                                     AveragePooling2D, 
                                     AveragePooling3D, 
                                     Flatten, 
                                     LeakyReLU
                                    )
    from tensorflow.keras.layers import (Dropout, BatchNormalization, 
                                    ELU, MaxPooling2D, MaxPooling3D, ActivityRegularization)
    from tensorflow.keras.layers import SeparableConv2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K
except ImportError:
    print('Tensorflow is not installed. Moving on')

    
import numpy as np

MLPRegressor = None
'''
class MLPRegressor(KerasRegressor):

    def __init__(
        self,
        initial_hidden_layer_size = 128,
        optimizer="adam",
        loss='mse', 
        optimizer__learning_rate=0.001,
        layer_size_decay_rate = 0.75,
        num_layers = 3, 
        activation='leaky_relu',
        batch_norm = True,
        l1_weight=0.001,
        l2_weight=0.01, 
        dropout_rate=0.1,
        epochs=200,
        verbose=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.hidden_layer_sizes = self._hidden_layer_sizes(initial_hidden_layer_size, 
                                                           num_layers, layer_size_decay_rate)
        self.batch_norm = batch_norm
        self.activation = activation 
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight 
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.verbose = verbose

    def _hidden_layer_sizes(self, initial_size, num_layers, decay_rate):
        """
        Calculate hidden layer sizes using exponential decay.
    
        :param initial_size: Size of the first hidden layer.
        :param num_layers: Total number of hidden layers.
        :param decay_rate: Rate of decay for layer sizes.
        :return: List of sizes for each hidden layer.
        """
        return (int(initial_size * np.exp(-decay_rate * i)) for i in range(num_layers))

    def _get_regularization_layer(self,  l1_weight, l2_weight ):
        """ Creates a regularization object.
        """
        return keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)
    
    def _get_activation_layer(self, function_name, alpha_parameter=0.2): 
        """ Creates an activation layer. 
        :param function name: Name of activation function (must be accepted by
                        `_check_activation_function`).
        :param alpha_parameter: Slope (used only for eLU and leaky ReLU functions).
        :return: layer_object: Instance of `keras.layers.Activation`,
                        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
        """
        if function_name == 'elu': 
            return ELU( alpha = alpha_parameter )
        if function_name == 'leaky_relu': 
            return LeakyReLU( alpha = alpha_parameter) 
        return Activation(function_name)  
    
    def _get_dropout_layer(self):
        """ Create a dropout object for the dense layers
        """
        return Dropout( rate = self.dropout_rate )
    
    def _get_batch_norm_layer( self ):
            """Creates batch-normalization layer.

            :return: layer_object: Instance of `keras.layers.BatchNormalization`.
            """
            return BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
    
    def _get_dense_layer(self, num_neurons, l1_weight, l2_weight, dense_bias='zeros',
                        kernel_initializer = 'glorot_uniform', output_layer=False): 
        """ Create a Dense layer with optionally regularization. 
        """
        return Dense( num_neurons ,
                              kernel_initializer = kernel_initializer,
                              use_bias           = True,
                              bias_initializer   = dense_bias,
                              activation         = None,
                              kernel_regularizer = self._get_regularization_layer( l1_weight, l2_weight) )
    
    
    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        
        self.model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_))
        self.model.add(inp)
        for n in self.hidden_layer_sizes:
            # Apply Dense layer with regularization 
            dense_layer = self._get_dense_layer(n, self.l1_weight, self.l2_weight)
            self.model.add(dense_layer)
            
            # Apply activation 
            activation_layer = self._get_activation_layer( 
                    function_name = self.activation )
            self.model.add(activation_layer)
        
            # Apply batch normalization (optional) 
            if self.batch_norm:
                batch_norm_layer= self._get_batch_norm_layer()
                self.model.add(batch_norm_layer)
            
            if self.dropout_rate > 0 : 
                dropout_layer = self._get_dropout_layer()
                self.model.add(dropout_layer)
            
        # Add the final layer. 
        out = keras.layers.Dense(1)
        self.model.add(out)
        
        # Compile the model.
        self.model.compile(optimizer = self.optimizer, 
                           loss = self.loss)

        if self.verbose > 0:
            print(self.model.summary())

        return self.model
'''

import numpy as np
import xgboost as xgb
from typing import Tuple

def weighted_mse_loss(preds, dtrain):
    # Extract the true labels and the weights from dtrain
    labels = dtrain.get_label()
    weights = dtrain.get_weight()  # Ensure that dtrain has weights set

    # Compute the error
    errors = labels - preds

    # Gradient and hessian
    grad = -2 * errors * weights
    hess = 2 * weights

    return grad, hess
    
    
ml_config = {
    
    "ElasticNet": {
        "model": ElasticNet,
        "search_space": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            "l1_ratio": [0, 0.0001, 0.001, 0.01, 0.1, 1.0],
        },
        "n_jobs": 60,
        "n_iter": 100,
        "patience": 20
    },
    
    
    "XGBRegressor": {
        "model": XGBRegressor,
        "model_params": {
            "seed": 123,
            "tree_method": "gpu_hist",
            "gpu_id": 0,
            "sampling_method": "gradient_based", 
            #"objective" : 'reg:squaredlogerror'
        },
        "search_space": {},
        "n_jobs": 1,
        "n_iter": 100,
        "patience": 20
    },
    
    
    "RFClassifier": {
        "model": RandomForestClassifier,
        "model_params": {
            "n_jobs": -1,
            "random_state": 123
        },
        "search_space": {
            "criterion": ["gini", "entropy"],
            "n_estimators": [100, 125, 150, 175, 200, 225, 250, 275, 300],
            "max_depth": [2, 5, 10, 15, 25, 40, None],
            "min_samples_split": [2, 5, 10, 15, 40],
            "min_samples_leaf": [2, 4, 5, 10, 15, 20, 25, 50],
            "class_weight": ["balanced", None]
        },
        "n_jobs": 1,
        "n_iter": 100,
        "patience": 20
    },
    
    
    "LogisticRegression": {
        "model": LogisticRegression,
        "model_params": {
            "penalty": "elasticnet",
            "solver": "saga",
            "random_state": 123
        },
        "search_space": {
            "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            "class_weight": [None, "balanced"]
        },
        "n_jobs": 60,
        "n_iter": 100,
        "patience": 20
    },
    
    
    "BaselineLR": {
        "model": LogisticRegression,
        "model_params": {
            "penalty": "elasticnet",
            "solver": "saga",
            "random_state": 123
        },
        "search_space": {
            "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            "class_weight": [None, "balanced"]
        },
        "n_jobs": 60,
        "n_iter": 100,
        "patience": 20
    },
    
    "XGBClassifier": {
        "model": XGBClassifier,
        "model_params": {
            "objective": "binary:logistic",
            "seed": 123,
            "tree_method": "gpu_hist",
            "gpu_id": 0
        },
        "search_space": {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3,5,7,9],
            "subsample": [0.5, 0.8, 1.0],
            "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "lambda": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
            "sampling_method": ["uniform", "gradient_based"],
            "min_child_weight": [1,3,5,7],
        },
        "n_jobs": 1,
        "n_iter": 100,
        "patience": 5
    },
    
    "NNRegressor": {
        "model": MLPRegressor,
        "model_params": {
            "epochs": 20
        },
        "search_space": {
            'initial_hidden_layer_size' : [32,64,128,256],
            "num_layers" : [1,2,3], 
            "activation" : ['elu', 'leaky_relu'],
            "batch_norm" : [True, False],
            "l1_weight" : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "l2_weight": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "dropout_rate": [0, 0.1, 0.3],
            "loss" : ['mae', 'mse']
        },
        "n_jobs": 1,
        "n_iter": 10,
        "patience": 5
    },
    
    
    "ExplainableBoostingRegressor": {
        "model": ExplainableBoostingRegressor,
        "model_params": {
            "random_state": 142
        },
        "search_space": {}, 
        "n_jobs": 1,
        "n_iter": 100,
        "patience": 20
    }
}

class MLConfiguration:
    @classmethod
    def get_config(self, model_name, X=None):
    
        results = ml_config.get(model_name, None)
        if results is None:
            raise KeyError(f'{model_name} does not have an existing configuration')
    
        if model_name in ['RandomForestClassifier', 'RandomForestRegressor']:
            results['search_space']['max_features'] = list(np.arange(1, n_features))
    
        return results
    
    @classmethod
    def get_scorer(self, target_type):
        
        if target_type == 'classification':
        
            def scorer(estimator, X, y):
                pred = estimator.predict_proba(X)[:,1]
                return -average_precision_score(y, pred)
        else:
            def scorer(estimator, X, y):
                pred = estimator.predict(X)
                return mean_squared_error(y, pred)
            
        return scorer 
