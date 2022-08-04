"""Calibration of predicted probabilities."""

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Balazs Kegl <balazs.kegl@gmail.com>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

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

def _fit(estimator, X, y): 
    return estimator.fit(X, y)
                
def _predict(estimator,X,y):
    return (estimator.predict_proba(X)[:,1], y)

class CalibratedClassifierCV(BaseEstimator, ClassifierMixin,
                             MetaEstimatorMixin):
    """Probability calibration with isotonic regression or sigmoid.

    See glossary entry for :term:`cross-validation estimator`.

    With this class, the base_estimator is fit on the train set of the
    cross-validation generator and the test set is used for calibration.
    The probabilities for each of the folds are then averaged
    for prediction. In case that cv="prefit" is passed to __init__,
    it is assumed that base_estimator has been fitted already and all
    data is used for calibration. Note that data for fitting the
    classifier and for calibrating it must be disjoint.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv=prefit, the
        classifier must have been fit already on data.

    method : 'sigmoid' or 'isotonic'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach. It is not advised to use isotonic calibration
        with too few calibration samples ``(<<1000)`` since it tends to
        overfit.
        Use sigmoids (Platt's calibration) in this case.

    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`sklearn.model_selection.KFold`
        is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    Attributes
    ----------
    classes_ : array, shape (n_classes)
        The class labels.

    calibrated_classifiers_ : list (len() equal to cv or 1 if cv == "prefit")
        The list of calibrated classifiers, one for each crossvalidation fold,
        which has been fitted on all but the validation fold and calibrated
        on the validation fold.

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """
    def __init__(self, base_estimator=None, method='isotonic', cv=None, n_jobs=1):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit the calibrated model

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

        self.calibrated_classifiers_ = []
        
        cv = check_cv(self.cv, y, classifier=True)
        fit_parameters = signature(self.base_estimator.fit).parameters
        estimator_name = type(self.base_estimator).__name__
        
        # Restructured to match the method for Platt (1999). Train an
        # estimator per fold. Collect the predictions into a single list
        # Train the calibration model. 
            
        parallel = Parallel(n_jobs=self.n_jobs)

        self.fit_estimators_ = parallel(delayed(
                _fit)(clone(self.base_estimator),X[train], y[train]) for train, _ in cv.split(X,y))
            
        results = parallel(delayed(
                _predict)(estimator , X[test], y[test]) for estimator, (_, test) in zip(self.fit_estimators_, cv.split(X,y)))

        cv_predictions = [item[0] for item in results ]
        cv_targets = [item[1] for item in results ]

        cv_predictions =  list(itertools.chain.from_iterable(cv_predictions)) 
        cv_targets =  list(itertools.chain.from_iterable(cv_targets))
            
        this_estimator = clone(self.base_estimator) 

        # Re-fit base_estimator on the whole dataset
        refit_estimator = this_estimator.fit(X,y) 

        calibrated_classifier = _CalibratedClassifier(
                    refit_estimator, method=self.method,
                    classes=self.classes_)
        
        # Fit the isotonic regression model. 
        calibrated_classifier.fit(cv_predictions, cv_targets)
        self.calibrated_classifiers_.append(calibrated_classifier)

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

        calibrated_classifier = self.calibrated_classifiers_[0]
        return calibrated_classifier.predict_proba(X)

    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _CalibratedClassifier:
    """Probability calibration with isotonic regression or sigmoid.

    It assumes that base_estimator has already been fit, and trains the
    calibration on the input set of the fit function. Note that this class
    should not be used as an estimator directly. Use CalibratedClassifierCV
    with cv="prefit" instead.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. No default value since
        it has to be an already fitted estimator.

    method : 'sigmoid' | 'isotonic'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach based on isotonic regression.

    classes : array-like, shape (n_classes,), optional
            Contains unique classes used to fit the base estimator.
            if None, then classes is extracted from the given target values
            in fit().

    See also
    --------
    CalibratedClassifierCV

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """
    def __init__(self, base_estimator, method='isotonic', classes=None):
        self.base_estimator = base_estimator
        self.method = method
        self.classes = classes

    def _preproc(self, X):
        n_classes = len(self.classes_)
        probabilities = self.base_estimator.predict_proba(X)[:,1]
        idx_pos_class = self.label_encoder_.\
            transform(self.base_estimator.classes_)

        return probabilities, idx_pos_class

    def fit(self, X, y):
        """Calibrate the fitted model

        Parameters
        ----------
        X : array-lie, shape (n_samples,)
            Predictions from the base_estimator

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.label_encoder_ = LabelEncoder()
        if self.classes is None:
            self.label_encoder_.fit(y)
        else:
            self.label_encoder_.fit(self.classes)

        self.classes_ = self.label_encoder_.classes_
        self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
        self.calibrator_.fit(X, y)

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
            The predicted probas. Can be exact zeros.
        """
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))

        probabilities, idx_pos_class = self._preproc(X)

        proba[:, 1] = self.calibrator_.predict(probabilities)

        # Normalize the probabilities
        if n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            proba /= np.sum(proba, axis=1)[:, np.newaxis]

        # XXX : for some reason all probas can be 0
        proba[np.isnan(proba)] = 1. / n_classes

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba

