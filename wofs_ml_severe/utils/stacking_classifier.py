"""Fit a stacking classifier from an ensemble of estimators"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import check_cv

def _fit(estimator, X, y):
    return estimator.fit(X, y)

def _predict(estimator,X,y):
    return (estimator.predict_proba(X)[:,1], y)


class StackingClassifier(BaseEstimator, ClassifierMixin,
                             MetaEstimatorMixin):
    """
    Stack ensemble of ML models and train a final estimator 
    on the predictions made by those models.
    """
    def __init__(self, estimators=None, cv=None, n_jobs=1):
        self.estimators = estimators
        self.cv = cv
        self.n_jobs = n_jobs

    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the stacked classifier.
        """
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False, allow_nd=True)
        X, y = indexable(X, y)
        le = LabelBinarizer().fit(y)
        self.classes_ = le.classes_

        # Check that each cross-validation fold can have at least one
        # example per class
        n_folds = self.cv if isinstance(self.cv, int) \
            else self.cv.n_folds if hasattr(self.cv, "n_folds") else None
        if n_folds and \
                np.any([np.sum(y == class_) < n_folds for class_ in
                        self.classes_]):
            raise ValueError("Requesting %d-fold cross-validation but provided"
                             " less than %d examples for at least one class."
                             % (n_folds, n_folds))

        cv = check_cv(self.cv, y, classifier=True)
        # Restructured to match the method for Platt (1999). Train an
        # estimator per fold. Collect the predictions into a single list
        # Train the calibration model. 

        parallel = Parallel(n_jobs=1)

        # Fit each estimator to the training portion of the cross-validation folds 
        self.fit_estimators_ = [parallel(delayed(
                _fit)(clone(estimator),X[train], y[train]) for train, _ in cv.split(X,y)) for estimator in self.estimators]

        all_X = []
        for estimator_set in self.fit_estimators_:
            parallel = Parallel(n_jobs=self.n_jobs)
            results = parallel(delayed(
                _predict)(estimator , X[test], y[test]) for estimator, (_, test) in zip(estimator_set, cv.split(X,y)))

            cv_predictions = [item[0] for item in results ]
            cv_targets = [item[1] for item in results ]

            cv_predictions =  list(itertools.chain.from_iterable(cv_predictions))
            cv_targets =  list(itertools.chain.from_iterable(cv_targets))

            all_X.append(cv_predictions)
        
        # Re-fit base_estimator on the whole dataset
        self.fit_estimators = [_fit(clone(estimator),X,y) for estimator in self.estimators]

        # Fit the final estimator based on the stacked ML models
        self.final_estimator = LogisticRegression()
        all_X = np.vstack(all_X)

        self.final_estimator.fit(all_X.T, cv_targets) 

        return self 

    def predict_proba(self, X):
        """Posterior probabilities of classification

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
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csc', 'csr', 'coo'],
                        force_all_finite=False)

        predictions = np.array([estimator.predict_proba(X)[:,1] for estimator in self.fit_estimators]).T

        final_predictions = self.final_estimator.predict_proba(predictions)

        return final_predictions
    










