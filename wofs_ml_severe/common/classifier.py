from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

"""
import keras
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, Flatten, LeakyReLU, ReLU
from keras.layers import Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
"""

import warnings
from inspect import signature

from math import log
import numpy as np
from joblib import delayed, Parallel
import itertools

from scipy.special import expit
from scipy.special import xlogy
from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import LabelEncoder

from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone,
                   MetaEstimatorMixin)
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from sklearn.utils.validation import _check_sample_weight
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import check_cv


class Classifier: 
    def __init__(model_name, n_jobs=25, **params):
        self._model_name = model_name
        self._n_jobs = n_jobs
        self._params = params
        
        return self._get_classifier()

    def _get_classifier( ):
        """Returns classifier machine learning model object."""
        # Random Forest Classifier 
        if  'RandomForestClassifier' in self._model_name:
            return RandomForestClassifier( n_jobs = njobs, criterion='entropy', random_state=42 )
        
        # Gradient-Boosted Tree Classifier 
        elif 'GradientBoostingClassifier' in self._model_name:
            return GradientBoostingClassifier()
        
        # Logistic Regression 
        elif 'LogisticRegression' in model_name:
            return LogisticRegression(n_jobs=njobs, solver='saga', penalty='elasticnet', max_iter=300, random_state=42)
        
        elif 'HistGradientBoosting' in model_name:
            try:
                categorical_features = self._params['categorical_features']
            except:
                raise KeyError('Expecting a categorical_features in params!') 
            return HistGradientBoostingClassifier(loss='binary_crossentropy', 
                                                  random_state=123, 
                                                  categorical_features=categorical_features)

        elif 'NeuralNetwork' in model_name:
            return MyNeuralNetworkClassifier()
        
        else:
            raise ValueError(f"{model_name} is not an accepted option!")


def calibration_model( classifier ):
    '''
    Returns the calibration machine learning model.

    usage: clf.fit( ) 
    '''
    return CalibratedClassifierCV(classifier, cv='prefit', method='isotonic')


class MyNeuralNetworkClassifier(BaseEstimator, ClassifierMixin,
                             MetaEstimatorMixin):
    """
    Build a dense neural network using Keras
    """
    def __init__(self, 
            choice={'layers':'two'}, 
            units1 = 512, 
            units2 = 128,
            activation='leaky_relu', 
            l1_weight=0.0,  
            l2_weight=0.0,
            use_batch_normalization=False,
            dropout_fraction=0.0,
            alpha_parameter=0.2,
            n_epochs=100,
            batch_size=1024,
            ):

        self.choice = choice
        self.units1 = units1
        self.units2 = units2
        self.activation=activation
        self.l1_weight=l1_weight
        self.l2_weight=l2_weight
        self.dropout_fraction=dropout_fraction
        self.alpha_parameter=alpha_parameter
        self.use_batch_normalization = use_batch_normalization
        self.n_epochs = n_epochs
        self.batch_size=batch_size

    def _build_the_network(self,):
        """
        Build the Keras Model
        """
        input_layer_object = self._get_input_layer(self.input_shape)
        last_layer_object = input_layer_object
   
        # Add the first dense layer 
        dense_layer_object = self._get_dense_layer( self.units1, self.l1_weight,  self.l2_weight )
        last_layer_object  = dense_layer_object( last_layer_object )

        activation_layer_object = self._get_activation_layer( 
            function_name =  self.activation, alpha_parameter=self.alpha_parameter )
        last_layer_object = activation_layer_object(last_layer_object)
            
        # Apply batch normalization (optional)
        if  self.use_batch_normalization:
            batch_norm_layer_object = self._get_batch_norm_layer()
            last_layer_object = batch_norm_layer_object(last_layer_object)
            
        # Apply weight dropout (optional)
        if self.dropout_fraction > 0:
            dropout_layer_object = self._get_dropout_layer(self.dropout_fraction)
            last_layer_object = dropout_layer_object(last_layer_object)

        # Add the second dense layer 
        dense_layer_object = self._get_dense_layer( self.units2, self.l1_weight,  self.l2_weight )
        last_layer_object  = dense_layer_object( last_layer_object )
            
        activation_layer_object = self._get_activation_layer(
            function_name =  self.activation, alpha_parameter=self.alpha_parameter )
        last_layer_object = activation_layer_object(last_layer_object)
            
        # Apply batch normalization (optional)
        if  self.use_batch_normalization:
            batch_norm_layer_object = self._get_batch_norm_layer()
            last_layer_object = batch_norm_layer_object(last_layer_object)
            
        # Apply weight dropout (optional)
        if self.dropout_fraction > 0:
            dropout_layer_object = self._get_dropout_layer(self.dropout_fraction)
            last_layer_object = dropout_layer_object(last_layer_object)
    
        if self.choice['layers'] == 'three':
            # Add the third dense layer 
            dense_layer_object = self._get_dense_layer( self.choice['units3'], 
                                                       self.l1_weight,  self.l2_weight )
            last_layer_object  = dense_layer_object( last_layer_object )
            
            activation_layer_object = self._get_activation_layer( 
                function_name =  self.activation, alpha_parameter=self.alpha_parameter )
            last_layer_object = activation_layer_object(last_layer_object)
            
            # Apply batch normalization (optional)
            if  self.use_batch_normalization:
                batch_norm_layer_object = self._get_batch_norm_layer()
                last_layer_object = batch_norm_layer_object(last_layer_object)
            
            # Apply weight dropout (optional)
            if self.dropout_fraction > 0:
                dropout_layer_object = self._get_dropout_layer(self.dropout_fraction)
                last_layer_object = dropout_layer_object(last_layer_object)
        
        # Add the final, output layer
        dense_layer_object = self._get_dense_layer( 1, 0.0, 0.0)
        last_layer_object  = dense_layer_object( last_layer_object )
        activation_layer_object = self._get_activation_layer(function_name='sigmoid')
        last_layer_object = activation_layer_object(last_layer_object)

        estimator = Model( inputs=input_layer_object, outputs=last_layer_object)
        estimator.compile(optimizer = Adam( ), loss = "binary_crossentropy")

        #print(estimator.summary())

        self.estimator_ = estimator


    def fit(self, X,y, params={}):
        """Fit the neural network

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False, allow_nd=True)
        X, y = indexable(X, y)
        le = LabelBinarizer().fit(y)
        self.classes_ = le.classes_

        self.input_shape = X.shape[1:]
        self._build_the_network()

        model_training_hist = self.estimator_.fit(X,y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=0)

        self.model_training_hist =  model_training_hist

        return self 

    def predict_proba(self, X):
        """Posterior probabilities of the neural network

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        X = check_array(X, accept_sparse=['csc', 'csr', 'coo'],
                        force_all_finite=False)
        
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))

        proba[:, 1] = self.estimator_.predict(X)[:,0]
        
        return proba

    def _get_input_layer( self, input_shape ):
        """ Creates the input layer.
        """
        return Input( shape = input_shape )

    def _get_activation_layer(self, function_name, alpha_parameter=0.2):
        """ Creates an activation layer. 
        :param function name: Name of activation function (must be accepted by
                        `_check_activation_function`).
        :param alpha_parameter: Slope (used only for eLU and leaky ReLU functions).
        :return: layer_object: Instance of `keras.layers.Activation`,
                        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
        """
        if function_name == 'relu':
            return ReLU( )
        elif function_name == 'leaky_relu':
            return LeakyReLU( alpha = alpha_parameter)
        
        return Activation(function_name)


    def _get_dense_layer( self, num_neurons, l1_weight, l2_weight ):
        """ Create a Dense layer with optionally regularization. 
        """
        return Dense( num_neurons ,
                              kernel_initializer = 'glorot_uniform',
                              use_bias           = True,
                              bias_initializer   = 'zeros',
                              activation         = None,
                              kernel_regularizer = self._get_regularization_layer( l1_weight, l2_weight) )


    def _get_regularization_layer(self,  l1_weight, l2_weight ):
        """ Creates a regularization object.
        """
        return keras.regularizers.l1_l2(l1=l1_weight, l2=l2_weight)


    def _get_batch_norm_layer( self ):
            """Creates batch-normalization layer.

            :return: layer_object: Instance of `keras.layers.BatchNormalization`.
            """
            return BatchNormalization( axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)

    def _get_dropout_layer( self, dropout_fraction ):
        """ Create a dropout object for the dense layers
        """
        return Dropout( rate = dropout_fraction )






