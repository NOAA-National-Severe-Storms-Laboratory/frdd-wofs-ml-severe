import shap
#import sys, os
#sys.path.insert(0, '/home/monte.flora/python_packages/scikit-explain')

import skexplain 
import pandas as pd 
import numpy as np 

from ..common.util import fix_data

class LocalExplainer:
    """Used to create the set of local top predictors and their values 
    for the interactive cbWoFS explainability graphics."""
    
    def __init__(self, model, X, X_train=None, n_pred = 5):
        self._model = model
        self._X = X 
        self._features = X.columns
        self._X_train = X_train
        self._n_pred = n_pred

    def top_features(self, target, method='coefs'):
        """Returns the top predictors and their values for a set of input data.
        Parameters
        -----------------
        method  : 'coefs', 'shap'
            If 'coefs' (default method), use the logistic regression coefficients to 
            determine the top predictors. If 'shap', use the SHAP permutation method to 
            determine the top predictors. 
        """
        if method == 'coefs':
            inputs = self.lr_inputs()
            attrs = self.to_dataframe(inputs, self._features)
            func = np.product

        elif method == 'shap':
            print('Using the SHAP method...')
            contrib_ds = self._shap()
            attrs = pd.DataFrame(contrib_ds['shap_values__Model'].values, 
                         columns = contrib_ds.attrs['features'])
            func = np.sum
        else:
            raise ValueError('method must be "shap" or "coefs!"')
            
        n_examples = attrs.shape[0]   
        results = [self._sort_attributions(func, attrs.iloc[i,:], i) for i in range(n_examples)]
    
        top_features = [r[0] for r in results]
        # The lists are nested. 
        top_features = [[f'{f}_{target}' for f in lst] for lst in top_features] 
        top_values = np.array([r[1] for r in results])     
        
        return top_features, top_values
        
    def to_dataframe(self, attrs, features ):
        return pd.DataFrame(attrs, columns = features)
           

    def just_transforms(self, estimator):
        """Applies all transforms to the data, without applying last 
        estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        X = fix_data(self._X)
    
        Xt = X
        for name, transform in estimator.steps[:-1]:
            Xt = transform.transform(Xt)
        return Xt


    def lr_inputs(self):
        """Compute the product of the model coefficients and processed inputs (e.g., scaling)."""
        # Scale the inputs. 
        try:
            base_est = self._model.estimators[0].calibrated_classifiers_[0].base_estimator
        except:
            base_est = self._model.calibrated_classifiers_[0].base_estimator
    
        Xt = self.just_transforms(base_est)
        # Get the model coefficients. 
        coef = base_est.named_steps['model'].coef_[0,:]
    
        inputs = np.exp(coef*Xt)
    
        return inputs

    def get_single_feature(self, features):
        for f in features:
            if 'amp_ens_mean' in f:
                return f
            elif 'ens_mean' in f:
                return f 
            else:
                return f 

    def _sort_attributions(self, func, attributions, ind):
    
        """Get the top features and their values for a single example. Sort the features 
        based on their attributions. """
    
        unique_features = np.unique([f.split('__')[0] for f in self._features])
        unique_attrs = np.zeros(len(unique_features))

        _unique_features = []
        for i, this_feature in enumerate(unique_features):
            these_features = [f for f in self._features if this_feature in f ]
        
            _unique_features.append(self.get_single_feature(these_features))
            val = func(attributions[these_features].values)
    
            unique_attrs[i] = val
    
        # Get the top predictors based on attribution. 
        inds = np.argsort(unique_attrs)[::-1][:self._n_pred]
    
        features_sorted = np.array(_unique_features)[inds]
        feature_values = self._X[features_sorted].values[ind,:]
    
        return features_sorted, feature_values
    
    
    def _shap(self):

        shap_kws={'masker' : shap.maskers.Partition(self._X_train, 
                                                    max_samples=100, 
                                                    clustering="correlation"), 
                                                   'algorithm' : 'permutation'}

        #try:
        #    base_est = self._model.estimators[0].calibrated_classifiers_[0].base_estimator
        #except:
        #    base_est = self._model.calibrated_classifiers_[0].base_estimator
        
        base_est = self._model
        
        explainer = skexplain.ExplainToolkit(('Model', base_est), X=self._X)
        contrib_ds = explainer.local_attributions(method='shap', 
                                           shap_kws = shap_kws,
                                          )
        return contrib_ds